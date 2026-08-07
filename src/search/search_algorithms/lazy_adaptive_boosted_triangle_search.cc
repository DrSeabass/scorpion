#include "lazy_adaptive_boosted_triangle_search.h"

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
#include <set>

using namespace std;

namespace lazy_adaptive_boosted_triangle_search {

LazyAdaptiveBoostedTriangleSearch::LazyAdaptiveBoostedTriangleSearch(
    const vector<shared_ptr<Evaluator>> &evals,
    bool reopen_closed,
    bool anytime,
    Schedule schedule,
    int credit_boost,
    const vector<shared_ptr<Evaluator>> &preferred_evals,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      schedule(schedule),
      credit_boost(credit_boost),
      evals(evals),
      num_lists(static_cast<int>(evals.size())),
      preferred_evals(preferred_evals),
      num_preferred(static_cast<int>(preferred_evals.size())),
      total_lists(static_cast<int>(evals.size()) + static_cast<int>(preferred_evals.size())),
      pruning_method(pruning),
      root_pending(true) {
    if (evals.empty()) {
        cerr << "LazyAdaptiveBoostedTriangleSearch: at least one guidance evaluator is required." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    preferred_source_index.reserve(preferred_evals.size());
    for (const shared_ptr<Evaluator> &pref_eval : preferred_evals) {
        int source = -1;
        for (int k = 0; k < num_lists; ++k) {
            if (evals[k].get() == pref_eval.get()) {
                source = k;
                break;
            }
        }
        if (source == -1) {
            cerr << "LazyAdaptiveBoostedTriangleSearch: every preferred_evals entry "
                    "must also appear in evals." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        preferred_source_index.push_back(source);
    }
    open_list_factories.reserve(total_lists);
    for (const shared_ptr<Evaluator> &e : evals) {
        open_list_factories.push_back(
            make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(e, false));
    }
    for (int source : preferred_source_index) {
        open_list_factories.push_back(
            make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(evals[source], false));
    }
}

void LazyAdaptiveBoostedTriangleSearch::initialize() {
    log << "Conducting lazy adaptive-boosted triangle search with "
        << num_lists << " guidance heuristic(s)"
        << ", schedule = " << (schedule == Schedule::SWEEP ? "sweep" : "pop")
        << ", credit_boost = " << credit_boost
        << ", " << num_preferred << " preferred-operator (helpful) list(s)"
        << " (" << total_lists << " list(s)/layer)"
        << ", (real) bound = " << bound << endl;

    assert(!evals.empty());

    set<Evaluator *> path_dependent;
    for (const shared_ptr<Evaluator> &e : evals)
        e->get_path_dependent_evaluators(path_dependent);
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

void LazyAdaptiveBoostedTriangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();

    if (log.is_at_least_debug()) {
        for (size_t i = 0; i < layers.size(); ++i) {
            log << "Lazy adaptive-boosted triangle layer " << (depth_offset + static_cast<int>(i))
                << " final budgets:";
            for (int k = 0; k < total_lists; ++k)
                log << " " << layers[i].progress[k].budget;
            log << endl;
        }
    }
}

void LazyAdaptiveBoostedTriangleSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(evals[0].get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

bool LazyAdaptiveBoostedTriangleSearch::has_non_empty_lists() const {
    if (root_pending)
        return true;
    for (size_t idx = 0; idx < layers.size(); ++idx) {
        if (!layer_empty(layers[idx], depth_offset + static_cast<int>(idx)))
            return true;
    }
    return false;
}

bool LazyAdaptiveBoostedTriangleSearch::layer_empty(const Layer &layer, int absolute_depth) const {
    // A budget-rejected edge sits in the single-slot pending buffer, not in
    // its EdgeOpenList (which only supports destructive remove_min()) --
    // see the class comment for why. Treat its (depth, list) as non-empty
    // regardless of what the underlying lists report.
    if (pending_operator_id != OperatorID::no_operator &&
        pending_absolute_depth == absolute_depth)
        return false;
    for (const unique_ptr<EdgeOpenList> &list : layer.lists) {
        if (!list->empty())
            return false;
    }
    return true;
}

LazyAdaptiveBoostedTriangleSearch::Layer LazyAdaptiveBoostedTriangleSearch::create_layer() const {
    vector<unique_ptr<EdgeOpenList>> lists;
    lists.reserve(total_lists);
    for (int k = 0; k < total_lists; ++k) {
        lists.push_back(open_list_factories[k]->create_edge_open_list());
    }
    return Layer(std::move(lists));
}

void LazyAdaptiveBoostedTriangleSearch::extend_layers(int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(create_layer());
    }
}

void LazyAdaptiveBoostedTriangleSearch::trim_empty_layers() {
    while (!layers.empty() && layer_empty(layers.front(), depth_offset)) {
        layers.pop_front();
        ++depth_offset;
    }
    while (!layers.empty() &&
           layer_empty(layers.back(), depth_offset + static_cast<int>(layers.size()) - 1)) {
        layers.pop_back();
    }
}

int LazyAdaptiveBoostedTriangleSearch::select_credit_served(
    const Layer &layer, int round_robin_served) const {
    int best = -1;
    for (int k = 0; k < total_lists; ++k) {
        if (layer.lists[k]->empty())
            continue;
        if (best == -1 || layer.progress[k].budget > layer.progress[best].budget)
            best = k;
    }
    if (best == -1)
        return round_robin_served;
    if (!layer.lists[round_robin_served]->empty() &&
        layer.progress[round_robin_served].budget == layer.progress[best].budget)
        return round_robin_served;
    return best;
}

int LazyAdaptiveBoostedTriangleSearch::select_round_robin_served(
    const Layer &layer, int round_robin_served) const {
    for (int offset = 0; offset < total_lists; ++offset) {
        int k = (round_robin_served + offset) % total_lists;
        if (!layer.lists[k]->empty())
            return k;
    }
    return round_robin_served;
}

void LazyAdaptiveBoostedTriangleSearch::record_expansion_credit(
    HeuristicProgress &hp, int h) const {
    if (hp.have_last_h && h < hp.last_h)
        hp.budget += credit_boost;
    hp.have_last_h = true;
    hp.last_h = h;
    --hp.budget;
}

void LazyAdaptiveBoostedTriangleSearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        log << "LazyAdaptiveBoostedTriangleSearch: improved incumbent with cost " << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

LazyAdaptiveBoostedTriangleSearch::ExpansionOutcome LazyAdaptiveBoostedTriangleSearch::process_candidate(
    const State &state,
    StateID predecessor_id,
    OperatorID operator_id,
    int g,
    int real_g,
    int source_layer_index,
    bool is_root,
    vector<int> &h_out) {
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

    bool is_dead_end = false;
    h_out.assign(total_lists, 0);
    for (int k = 0; k < num_lists; ++k) {
        int h = eval_context.get_evaluator_value_or_infinity(evals[k].get());
        if (h == EvaluationResult::INFTY && evals[k]->dead_ends_are_reliable())
            is_dead_end = true;
        h_out[k] = h;
    }
    for (int j = 0; j < num_preferred; ++j)
        h_out[num_lists + j] = h_out[preferred_source_index[j]];
    if (is_dead_end) {
        node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return ExpansionOutcome::SKIPPED;
    }

    vector<vector<OperatorID>> preferred_ops(num_preferred);
    for (int j = 0; j < num_preferred; ++j)
        preferred_ops[j] = eval_context.get_preferred_operators(preferred_evals[j].get());

    if (is_root) {
        node.open_initial();
        start_evaluator_statistics(eval_context);
        print_initial_evaluator_values(eval_context);
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
    }

    if (task_properties::is_goal_state(task_proxy, state)) {
        update_incumbent(state);
        return anytime_search ? ExpansionOutcome::SKIPPED : ExpansionOutcome::SOLVED;
    }

    node.close();
    statistics.inc_expanded();

    if (search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(g);
    }

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

        int succ_g = g + get_adjusted_cost(op);
        EvaluationContext succ_eval_context(eval_context, succ_g, false, nullptr);
        for (int k = 0; k < num_lists; ++k) {
            layers[source_layer_index + 1].lists[k]->insert(
                succ_eval_context, make_pair(state.get_id(), op_id));
        }
        for (int j = 0; j < num_preferred; ++j) {
            const vector<OperatorID> &preferred = preferred_ops[j];
            if (find(preferred.begin(), preferred.end(), op_id) != preferred.end()) {
                layers[source_layer_index + 1].lists[num_lists + j]->insert(
                    succ_eval_context, make_pair(state.get_id(), op_id));
            }
        }
    }

    return ExpansionOutcome::EXPANDED;
}

SearchStatus LazyAdaptiveBoostedTriangleSearch::step() {
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

    // Adaptive per-step depth budget (ported as-is from
    // adaptive_triangle_search, always on). Unspent budget carries between
    // steps; each step starts with at least one unit.
    depth_budget = max(1, depth_budget);

    const int sweep_served = sweep_count % total_lists;

    // Adaptive depth-budget trend signal: a single step-scoped counter over
    // every expansion in the step, regardless of which guidance list served
    // it.
    int last_expanded_h = 0;
    bool have_last_h = false;
    // Number of new frontier layers this step has paid for so far; the
    // first extension of the step is always free.
    int layers_added = 0;

    if (root_pending) {
        root_pending = false;
        State initial_state = state_registry.get_initial_state();
        vector<int> h_out;
        ExpansionOutcome outcome = process_candidate(
            initial_state,
            StateID::no_state,
            OperatorID::no_operator,
            0, 0, 0, true, h_out);
        if (outcome == ExpansionOutcome::SOLVED)
            return SOLVED;
        // Root has no predecessor edge and doesn't touch depth_budget (its
        // layer already exists from initialize()); the cascade below picks
        // up from i=0, which is now empty, and continues into the root's
        // freshly-inserted successors at layer 1 within this same step.
    }

    for (int i = 0; ; ++i) {
        if (i >= static_cast<int>(layers.size()))
            break;

        const int round_robin_served =
            (schedule == Schedule::SWEEP) ? sweep_served : pop_count % total_lists;
        const int served =
            (credit_boost == 0)
                ? select_round_robin_served(layers[i], round_robin_served)
                : select_credit_served(layers[i], round_robin_served);

        StateID predecessor_id = StateID::no_state;
        OperatorID operator_id = OperatorID::no_operator;
        bool have_candidate = false;

        // Reuse a previously budget-rejected edge for this exact
        // layer/list combo before consulting the list itself -- see the
        // class comment on the pending-edge buffer.
        if (pending_operator_id != OperatorID::no_operator &&
            pending_absolute_depth == depth_offset + i &&
            pending_list_index == served) {
            predecessor_id = pending_predecessor_id;
            operator_id = pending_operator_id;
            pending_operator_id = OperatorID::no_operator;
            have_candidate = true;
        } else {
            EdgeOpenList &list = *layers[i].lists[served];
            while (!list.empty()) {
                EdgeOpenListEntry next = list.remove_min();
                StateID p_id = next.first;
                OperatorID op_id = next.second;
                if (p_id == StateID::no_state || op_id == OperatorID::no_operator)
                    continue;
                State predecessor_state = state_registry.lookup_state(p_id);
                OperatorProxy op = task_proxy.get_operators()[op_id];
                if (!task_properties::is_applicable(op, predecessor_state))
                    continue;
                // Full staleness gate mirroring process_candidate's own,
                // checked here (not just is_applicable) so depth_budget is
                // only ever spent on a genuinely expandable candidate.
                State succ_state = state_registry.get_successor_state(predecessor_state, op);
                SearchNode succ_node = search_space.get_node(succ_state);
                SearchNode predecessor_node = search_space.get_node(predecessor_state);
                int cand_g = predecessor_node.get_g() + get_adjusted_cost(op);
                bool cand_reopen = reopen_closed_nodes && succ_node.is_closed() &&
                    !succ_node.is_dead_end() && (cand_g < succ_node.get_g());
                if (!reopen_closed_nodes && !succ_node.is_new())
                    continue;
                if (!(succ_node.is_new() || cand_reopen))
                    continue;
                predecessor_id = p_id;
                operator_id = op_id;
                have_candidate = true;
                break;
            }
        }

        if (!have_candidate)
            continue;

        // We have a genuinely-expandable edge. Ensure layer i+1 exists to
        // receive its successors -- free if already in the deque,
        // otherwise pay one depth_budget unit. The first extension of the
        // step is granted unconditionally (even with a depleted/negative
        // budget); the gate is consulted only for extensions beyond that.
        if (i + 1 >= static_cast<int>(layers.size())) {
            if (layers_added > 0 && depth_budget <= 0) {
                // Can't afford it -- park this edge for the next time this
                // exact layer/list combo is served (see the pending-edge
                // buffer's class comment) and halt the cascade.
                pending_predecessor_id = predecessor_id;
                pending_operator_id = operator_id;
                pending_list_index = served;
                pending_absolute_depth = depth_offset + i;
                break;
            }
            --depth_budget;
            extend_layers(1);
            ++layers_added;
        }

        State predecessor = state_registry.lookup_state(predecessor_id);
        OperatorProxy op = task_proxy.get_operators()[operator_id];
        SearchNode predecessor_node = search_space.get_node(predecessor);
        int g = predecessor_node.get_g() + get_adjusted_cost(op);
        // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for NORMAL cost type.
        int real_g = predecessor_node.get_g() + op.get_cost();
        State state = state_registry.get_successor_state(predecessor, op);

        vector<int> h_out;
        ExpansionOutcome outcome = process_candidate(
            state, predecessor_id, operator_id, g, real_g, i, false, h_out);
        if (outcome == ExpansionOutcome::SOLVED)
            return SOLVED;

        if (outcome == ExpansionOutcome::EXPANDED) {
            if (credit_boost != 0)
                record_expansion_credit(layers[i].progress[served], h_out[served]);

            // Adaptive depth-budget trend signal (see above): an informed
            // transition (h decreases relative to the previous expansion
            // in this step) refunds one unit, an uninformed one debits one.
            if (have_last_h) {
                if (h_out[served] < last_expanded_h)
                    ++depth_budget;
                else
                    --depth_budget;
            }
            last_expanded_h = h_out[served];
            have_last_h = true;

            // POP schedule: advance the round-robin per expansion. No-op
            // for SWEEP.
            ++pop_count;
        }
    }

    trim_empty_layers();
    ++sweep_count;
    return IN_PROGRESS;
}

}
