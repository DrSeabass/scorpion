#include "lazy_triangle_lama_mimick_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../open_lists/best_first_open_list.h"
#include "../plan_manager.h"
#include "../pruning_method.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"

#include "../utils/logging.h"
#include "../utils/system.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>

using namespace std;

namespace lazy_triangle_lama_mimick_search {
// Enable only for targeted diagnostic runs. This removes CSV I/O and the
// per-depth accounting from performance experiments when false.
static constexpr bool ENABLE_TRIANGLE_LAMA_DIAGNOSTICS = false;

LazyTriangleLamaMimickSearch::LazyTriangleLamaMimickSearch(
    const vector<shared_ptr<Evaluator>> &evals,
    int slope,
    bool reopen_closed,
    bool anytime,
    int boost_amount,
    const vector<shared_ptr<Evaluator>> &preferred_evals,
    bool guide_by_pruning,
    const shared_ptr<Evaluator> &pruning_heuristic,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      slope(slope),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      boost_amount(boost_amount),
      evals(evals),
      num_lists(static_cast<int>(evals.size())),
      preferred_evals(preferred_evals),
      num_preferred(preferred_evals.empty() ? 0 : static_cast<int>(evals.size())),
      total_lists(
          static_cast<int>(evals.size()) + static_cast<int>(preferred_evals.size()) +
          (guide_by_pruning && pruning_heuristic != nullptr ? 1 : 0)),
      pruning_method(pruning),
      guide_by_pruning(guide_by_pruning),
      use_pruner_queue(guide_by_pruning && pruning_heuristic != nullptr),
      pruning_heuristic(pruning_heuristic),
      root_pending(true) {
    if (slope <= 0) {
        cerr << "LazyTriangleLamaMimickSearch: slope must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (evals.empty()) {
        cerr << "LazyTriangleLamaMimickSearch: at least one guidance evaluator is required." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    open_list_factories.reserve(total_lists);
    for (const shared_ptr<Evaluator> &e : evals) {
        open_list_factories.push_back(
            make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(e, false));
        if (num_preferred) {
            open_list_factories.push_back(
                make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(e, true));
        }
    }
    if (use_pruner_queue) {
        open_list_factories.push_back(
            make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(pruning_heuristic, false));
    }
}

int LazyTriangleLamaMimickSearch::guidance_index(int evaluator_index) const {
    return num_preferred ? 2 * evaluator_index : evaluator_index;
}

int LazyTriangleLamaMimickSearch::preferred_index(int evaluator_index) const {
    assert(num_preferred);
    return 2 * evaluator_index + 1;
}

void LazyTriangleLamaMimickSearch::initialize() {
    log << "Conducting lazy triangle-LAMA-mimick search with slope " << slope
        << ", " << num_lists << " guidance heuristic(s)"
        << ", boost_amount = " << boost_amount
        << ", " << num_preferred << " preferred-operator (helpful) list(s)"
        << ", guide_by_pruning = " << use_pruner_queue
        << " (" << total_lists << " list(s)/layer)"
        << ", (real) bound = " << bound << endl;

    priorities.assign(total_lists, 0);
    if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS) {
        diagnostic_file.open("triangle_lama_layer_diagnostics.csv");
        diagnostic_file
            << "snapshot_sweeps,depth,removals,stale_removals,expansions,"
               "layers_considered,nonempty_layers_considered,max_sweep_width\n";
        diagnostic_file.flush();
    }

    assert(!evals.empty());

    set<Evaluator *> path_dependent;
    for (const shared_ptr<Evaluator> &e : evals)
        e->get_path_dependent_evaluators(path_dependent);
    if (pruning_heuristic) {
        pruning_heuristic->get_path_dependent_evaluators(path_dependent);
    }
    path_dependent_evaluators.assign(path_dependent.begin(), path_dependent.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    layers.clear();
    layers.push_back(create_layer());
    root_pending = true;

    pruning_method->initialize(task);
}

void LazyTriangleLamaMimickSearch::write_diagnostic_snapshot() {
    if constexpr (!ENABLE_TRIANGLE_LAMA_DIAGNOSTICS)
        return;
    if (!diagnostic_file)
        return;
    for (const auto &[depth, values] : layer_diagnostics) {
        diagnostic_file << diagnostic_sweeps << ',' << depth << ','
            << values.removals << ',' << values.stale_removals << ','
            << values.expansions << ',' << diagnostic_layers_considered << ','
            << diagnostic_nonempty_layers_considered << ','
            << diagnostic_max_sweep_width << '\n';
    }
    diagnostic_file.flush();
}

void LazyTriangleLamaMimickSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();

    if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS) {
        log << "Triangle diagnostic sweeps: " << diagnostic_sweeps << endl;
        log << "Triangle diagnostic layers considered: "
            << diagnostic_layers_considered << endl;
        log << "Triangle diagnostic nonempty layers considered: "
            << diagnostic_nonempty_layers_considered << endl;
        log << "Triangle diagnostic maximum sweep width: "
            << diagnostic_max_sweep_width << endl;
        for (const auto &[depth, values] : layer_diagnostics) {
            log << "Triangle diagnostic layer " << depth
                << ": removals=" << values.removals
                << ", stale=" << values.stale_removals
                << ", expansions=" << values.expansions << endl;
        }
    }

    if (log.is_at_least_debug()) {
        log << "Lazy triangle-LAMA-mimick final priorities:";
        for (int k = 0; k < total_lists; ++k)
            log << " " << priorities[k];
        log << endl;
    }
}

void LazyTriangleLamaMimickSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(evals[0].get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

bool LazyTriangleLamaMimickSearch::has_non_empty_lists() const {
    if (root_pending)
        return true;
    for (const Layer &layer : layers) {
        if (!layer_empty(layer))
            return true;
    }
    return false;
}

bool LazyTriangleLamaMimickSearch::layer_empty(const Layer &layer) const {
    for (const unique_ptr<EdgeOpenList> &list : layer.lists) {
        if (!list->empty())
            return false;
    }
    return true;
}

LazyTriangleLamaMimickSearch::Layer LazyTriangleLamaMimickSearch::create_layer() const {
    vector<unique_ptr<EdgeOpenList>> lists;
    lists.reserve(total_lists);
    for (int k = 0; k < total_lists; ++k) {
        lists.push_back(open_list_factories[k]->create_edge_open_list());
    }
    return Layer(std::move(lists));
}

void LazyTriangleLamaMimickSearch::extend_layers(int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(create_layer());
    }
}

void LazyTriangleLamaMimickSearch::trim_empty_layers() {
    while (!layers.empty() && layer_empty(layers.front())) {
        layers.pop_front();
        ++depth_offset;
    }
    while (!layers.empty() && layer_empty(layers.back())) {
        layers.pop_back();
    }
}

int LazyTriangleLamaMimickSearch::select_served() const {
    int best = 0;
    for (int k = 1; k < total_lists; ++k) {
        if (priorities[k] < priorities[best])
            best = k;
    }
    return best;
}

int LazyTriangleLamaMimickSearch::select_available_served(
    const Layer &layer, int intended) const {
    for (int offset = 0; offset < total_lists; ++offset) {
        int k = (intended + offset) % total_lists;
        if (!layer.lists[k]->empty())
            return k;
    }
    return intended;
}

void LazyTriangleLamaMimickSearch::boost_preferred_lists() {
    for (int k = 0; k < num_preferred; ++k)
        priorities[preferred_index(k)] -= boost_amount;
}

void LazyTriangleLamaMimickSearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        log << "LazyTriangleLamaMimickSearch: improved incumbent with cost " << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

LazyTriangleLamaMimickSearch::ExpansionOutcome LazyTriangleLamaMimickSearch::process_candidate(
    const State &state,
    StateID predecessor_id,
    OperatorID operator_id,
    int g,
    int real_g,
    int source_layer_index,
    bool is_root) {
    SearchNode node = search_space.get_node(state);

    if (!reopen_closed_nodes && !node.is_new())
        return ExpansionOutcome::SKIPPED;

    bool reopen = reopen_closed_nodes && node.is_closed() && !node.is_dead_end() && (g < node.get_g());

    if (!(node.is_new() || reopen))
        return ExpansionOutcome::SKIPPED;

    if (!is_root) {
        assert(predecessor_id != StateID::no_state);
        assert(operator_id != OperatorID::no_operator);
        if (!path_dependent_evaluators.empty()) {
            State parent_state = state_registry.lookup_state(predecessor_id);
            for (Evaluator *evaluator : path_dependent_evaluators) {
                evaluator->notify_state_transition(parent_state, operator_id, state);
            }
        }
    }

    statistics.inc_evaluated_states();
    EvaluationContext eval_context(state, g, true, &statistics, num_preferred > 0);

    // Evaluate all N guidance heuristics now that the state is actually
    // being popped for expansion. A state is dead iff any reliable
    // heuristic proves it so.
    bool is_dead_end = false;
    for (int k = 0; k < num_lists; ++k) {
        int h = eval_context.get_evaluator_value_or_infinity(evals[k].get());
        if (h == EvaluationResult::INFTY && evals[k]->dead_ends_are_reliable())
            is_dead_end = true;
    }
    if (is_dead_end) {
        node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return ExpansionOutcome::SKIPPED;
    }

    // The admissible pruning_heuristic stays lazy exactly like the guidance
    // heuristics: evaluated here, once, only for a state that is actually
    // being popped for expansion -- never per generated successor. A state
    // exceeding the current bound is skipped without being marked a dead
    // end (it may still be reachable more cheaply via a different
    // predecessor later); one the heuristic reliably proves unsolvable is
    // marked dead end, exactly like the guidance heuristics above.
    if (pruning_heuristic) {
        int prune_h = eval_context.get_evaluator_value_or_infinity(pruning_heuristic.get());
        if (prune_h == EvaluationResult::INFTY) {
            if (pruning_heuristic->dead_ends_are_reliable()) {
                node.mark_as_dead_end();
                statistics.inc_dead_ends();
            }
            return ExpansionOutcome::SKIPPED;
        }
        if (bound != numeric_limits<int>::max() && real_g + prune_h >= bound)
            return ExpansionOutcome::SKIPPED;
    }

    vector<OperatorID> preferred_ops;
    for (const shared_ptr<Evaluator> &evaluator : preferred_evals) {
        for (OperatorID op_id : eval_context.get_preferred_operators(evaluator.get())) {
            if (find(preferred_ops.begin(), preferred_ops.end(), op_id) == preferred_ops.end())
                preferred_ops.push_back(op_id);
        }
    }

    // LAMA's real progress check/reward (see class comment): fires here, at
    // pop/expansion time -- the same moment real LazySearch::step() checks
    // it -- for every state actually evaluated, except the initial state's
    // own evaluation (matching real LAMA's own no-reward-from-root rule).
    if (is_root) {
        node.open_initial();
        real_g_values[state] = 0;
        start_evaluator_statistics(eval_context);
        print_initial_evaluator_values(eval_context);
        if (search_progress.check_progress(eval_context))
            statistics.print_checkpoint_line(g);
    } else {
        State parent_state = state_registry.lookup_state(predecessor_id);
        SearchNode parent_node = search_space.get_node(parent_state);
        OperatorProxy op = task_proxy.get_operators()[operator_id];
        if (reopen) {
            statistics.inc_reopened();
            node.reopen_closed_node(parent_node, op, get_adjusted_cost(op));
        } else {
            node.open_new_node(parent_node, op, get_adjusted_cost(op));
        }
        real_g_values[state] = real_g;
    }

    if (task_properties::is_goal_state(task_proxy, state)) {
        update_incumbent(state);
        return anytime_search ? ExpansionOutcome::SKIPPED : ExpansionOutcome::SOLVED;
    }

    // Match LazySearch::step(): the root reports progress but is not
    // rewarded, and non-root progress is rewarded only after the goal test.
    if (!is_root && search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(g);
        boost_preferred_lists();
    }

    node.close();
    statistics.inc_expanded();

    while (static_cast<int>(layers.size()) <= source_layer_index + 1) {
        layers.push_back(create_layer());
    }

    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(state, applicable_ops);
    pruning_method->prune_operators(state, applicable_ops);

    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        int succ_real_g = real_g + op.get_cost();
        if (succ_real_g >= bound)
            continue;

        statistics.inc_generated();

        // Rank the successor by the parent's already-known h, reusing the
        // parent's EvaluationContext cache -- lazy_triangle_search's and
        // FD's LazySearch's own idiom.
        int succ_g = g + get_adjusted_cost(op);
        EvaluationContext succ_eval_context(eval_context, succ_g, false, nullptr);
        // Guidance lists (and the optional pruner list, ranked by the same
        // parent-h proxy -- pruning_heuristic stays lazy just like the
        // guidance heuristics, see the class comment) always receive every
        // successor.
        for (int k = 0; k < num_lists; ++k) {
            layers[source_layer_index + 1].lists[guidance_index(k)]->insert(
                succ_eval_context, make_pair(state.get_id(), op_id));
        }
        if (use_pruner_queue) {
            layers[source_layer_index + 1].lists[total_lists - 1]->insert(
                succ_eval_context, make_pair(state.get_id(), op_id));
        }
        if (find(preferred_ops.begin(), preferred_ops.end(), op_id) != preferred_ops.end()) {
            EvaluationContext preferred_succ_eval_context(
                eval_context, succ_g, true, nullptr);
            for (int k = 0; k < num_preferred; ++k) {
                layers[source_layer_index + 1].lists[preferred_index(k)]->insert(
                    preferred_succ_eval_context, make_pair(state.get_id(), op_id));
            }
        }
    }

    return ExpansionOutcome::EXPANDED;
}

SearchStatus LazyTriangleLamaMimickSearch::step() {
    if (!anytime_search && found_solution())
        return SOLVED;

    if (!has_non_empty_lists()) {
        if (found_solution()) {
            log << "All open lists are empty -- best solution found." << endl;
            return SOLVED;
        }
        log << "All open lists are empty -- no solution!" << endl;
        return FAILED;
    }

    extend_layers(slope);

    // The served list for the whole dive is decided once here, before the
    // cascade loop -- AlternationOpenList::remove_min's selection rule:
    // lowest priority counter wins, ties keep the lowest index.
    const int locked_served = select_served();

    const int num_layers = static_cast<int>(layers.size());
    const int sweep_width = num_layers - 1;
    if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS) {
        ++diagnostic_sweeps;
        diagnostic_layers_considered += sweep_width;
        diagnostic_max_sweep_width = max(diagnostic_max_sweep_width, sweep_width);
    }
    for (int i = 0; i < num_layers - 1; ++i) {
        if (i == 0 && root_pending) {
            root_pending = false;
            State initial_state = state_registry.get_initial_state();
            ExpansionOutcome outcome = process_candidate(
                initial_state,
                StateID::no_state,
                OperatorID::no_operator,
                0, 0, i, true);
            if (outcome == ExpansionOutcome::SOLVED)
                return SOLVED;
            continue;
        }

        // The locked-in list can be selectively empty at this specific
        // layer once helpful lists exist -- fall back to the nearest
        // non-empty list.
        const int served = select_available_served(layers[i], locked_served);
        EdgeOpenList &list = *layers[i].lists[served];
        if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS)
            if (!list.empty())
                ++diagnostic_nonempty_layers_considered;
        bool expanded = false;
        while (!list.empty() && !expanded) {
            EdgeOpenListEntry next = list.remove_min();
            // AlternationOpenList charges every remove_min(), including an
            // edge that LazySearch subsequently discards as a duplicate.
            ++priorities[served];
            if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS)
                ++layer_diagnostics[depth_offset + i].removals;

            StateID predecessor_id = next.first;
            OperatorID operator_id = next.second;
            if (predecessor_id == StateID::no_state || operator_id == OperatorID::no_operator) {
                if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS)
                    ++layer_diagnostics[depth_offset + i].stale_removals;
                continue;
            }

            State predecessor = state_registry.lookup_state(predecessor_id);
            OperatorProxy op = task_proxy.get_operators()[operator_id];
            if (!task_properties::is_applicable(op, predecessor)) {
                if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS)
                    ++layer_diagnostics[depth_offset + i].stale_removals;
                continue;
            }

            SearchNode predecessor_node = search_space.get_node(predecessor);
            int g = predecessor_node.get_g() + get_adjusted_cost(op);
            int real_g = real_g_values[predecessor] + op.get_cost();
            State state = state_registry.get_successor_state(predecessor, op);

            ExpansionOutcome outcome = process_candidate(
                state, predecessor_id, operator_id, g, real_g, i, false);
            if (outcome == ExpansionOutcome::SOLVED)
                return SOLVED;
            expanded = (outcome == ExpansionOutcome::EXPANDED);
            if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS) {
                if (expanded)
                    ++layer_diagnostics[depth_offset + i].expansions;
                else
                    ++layer_diagnostics[depth_offset + i].stale_removals;
            }
        }
    }

    if constexpr (ENABLE_TRIANGLE_LAMA_DIAGNOSTICS)
        write_diagnostic_snapshot();

    trim_empty_layers();
    return IN_PROGRESS;
}

}
