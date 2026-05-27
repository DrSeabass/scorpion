#include "multi_triangle_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../evaluator.h"
#include "../plan_manager.h"
#include "../pruning_method.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"

#include "../utils/logging.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <set>

using namespace std;

namespace multi_triangle_search {

MultiTriangleSearch::MultiTriangleSearch(
    const vector<shared_ptr<Evaluator>> &evals,
    int slope,
    bool reopen_closed,
    bool anytime,
    Schedule schedule,
    bool guide_by_pruning,
    const shared_ptr<Evaluator> &pruning_heuristic,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      slope(slope),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      schedule(schedule),
      guide_by_pruning(guide_by_pruning),
      evals(evals),
      num_lists(static_cast<int>(evals.size())),
      use_pruner_queue(guide_by_pruning && pruning_heuristic != nullptr),
      total_lists(static_cast<int>(evals.size()) + (guide_by_pruning && pruning_heuristic != nullptr ? 1 : 0)),
      pruning_heuristic(pruning_heuristic),
      pruning_method(pruning) {
    if (slope <= 0) {
        cerr << "MultiTriangleSearch: slope must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (evals.empty()) {
        cerr << "MultiTriangleSearch: at least one guidance evaluator is required." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void MultiTriangleSearch::initialize() {
    log << "Conducting multi-heuristic triangle search with slope " << slope
        << ", " << num_lists << " guidance heuristic(s)"
        << ", schedule = " << (schedule == Schedule::SWEEP ? "sweep" : "pop")
        << ", guide_by_pruning = " << use_pruner_queue
        << " (" << total_lists << " list(s)/layer)"
        << ", (real) bound = " << bound << endl;

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

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    // Evaluate the initial state with all N guidance heuristics. It is a dead
    // end iff any reliable heuristic proves it so (multi-source dead-end).
    bool is_dead_end = false;
    vector<int> initial_h;
    initial_h.reserve(total_lists);
    for (int k = 0; k < num_lists; ++k) {
        int h = eval_context.get_evaluator_value_or_infinity(evals[k].get());
        if (h == EvaluationResult::INFTY && evals[k]->dead_ends_are_reliable())
            is_dead_end = true;
        initial_h.push_back(h);
    }
    if (use_pruner_queue) {
        // Seed the pruner queue from the admissible h of the initial state.
        int prune_h =
            eval_context.get_evaluator_value_or_infinity(pruning_heuristic.get());
        if (prune_h == EvaluationResult::INFTY &&
            pruning_heuristic->dead_ends_are_reliable())
            is_dead_end = true;
        initial_h.push_back(prune_h);
    }

    extend_open_lists(1);

    if (is_dead_end) {
        log << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context)) {
            statistics.print_checkpoint_line(0);
        }
        start_evaluator_statistics(eval_context);

        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();
        if (task_properties::is_goal_state(task_proxy, initial_state)) {
            update_incumbent(initial_state);
        } else {
            insert_successor(0, initial_state.get_id(), 0, initial_h);
        }
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void MultiTriangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

void MultiTriangleSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    // Report f-value progress against the primary guidance heuristic.
    int value = eval_context.get_evaluator_value_or_infinity(evals[0].get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

void MultiTriangleSearch::extend_open_lists(int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        open_lists.emplace_back(total_lists);
    }
}

bool MultiTriangleSearch::layer_empty(int layer) const {
    for (const OpenList &list : open_lists[layer]) {
        if (!list.empty())
            return false;
    }
    return true;
}

void MultiTriangleSearch::recompute_max_active_layer() {
    while (max_active_layer >= 0 && layer_empty(max_active_layer)) {
        --max_active_layer;
    }
}

void MultiTriangleSearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        log << "MultiTriangleSearch: improved incumbent with cost " << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

bool MultiTriangleSearch::evaluate_and_prepare_node(
    const State &state, SearchNode &node, int g,
    vector<int> &h_out, bool is_new_evaluation) {
    EvaluationContext eval_context(state, g, false, &statistics);
    if (is_new_evaluation)
        statistics.inc_evaluated_states();

    // Evaluate all N guidance heuristics from a single context, so the
    // path-dependent state is shared and check_progress sees every heuristic.
    // A state is dead iff any reliable heuristic proves it so.
    h_out.resize(num_lists);
    for (int k = 0; k < num_lists; ++k) {
        int h = eval_context.get_evaluator_value_or_infinity(evals[k].get());
        if (h == EvaluationResult::INFTY && evals[k]->dead_ends_are_reliable()) {
            node.mark_as_dead_end();
            statistics.inc_dead_ends();
            return false;
        }
        h_out[k] = h;
    }

    if (is_new_evaluation && search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(node.get_g());
    }

    return true;
}

void MultiTriangleSearch::insert_successor(
    int layer, StateID id, int g, const vector<int> &hs) {
    assert(layer >= 0);
    assert(static_cast<int>(hs.size()) == total_lists);
    if (layer >= static_cast<int>(open_lists.size())) {
        extend_open_lists(layer + 1 - static_cast<int>(open_lists.size()));
    }
    for (int k = 0; k < total_lists; ++k) {
        open_lists[layer][k].push({id, hs[k], g});
    }
    if (layer > max_active_layer)
        max_active_layer = layer;
}

SearchStatus MultiTriangleSearch::step() {
    while (!open_lists.empty() && layer_empty(0)) {
        open_lists.pop_front();
        --max_active_layer;
    }
    max_active_layer = max(max_active_layer, 0);

    if (open_lists.empty()) {
        if (found_solution()) {
            log << "All open lists are empty -- best solution found." << endl;
            return SOLVED;
        }
        log << "All open lists are empty -- no solution!" << endl;
        return FAILED;
    }

    // The cascade bound is the step-start max_active_layer plus slope; the
    // deque grows lazily as successors get inserted, rather than being
    // pre-extended by slope at the top of step().
    const int cascade_cap = max_active_layer + slope;

    // SWEEP: one guidance list owns the entire cascade dive this step; the
    // served index rotates between steps. POP: the served index advances per
    // expansion (below), so successive expansions down the dive alternate
    // heuristics. Because every live state is inserted into all N lists, the
    // served list being (live-)empty at a layer means the layer is live-empty
    // -- skipping it forfeits nothing.
    const int sweep_served = step_count % total_lists;

    for (int i = 0; i < cascade_cap; ++i) {
        // End the cascade as soon as we run off the end of the deque.
        if (i >= static_cast<int>(open_lists.size()))
            break;
        const int served =
            (schedule == Schedule::SWEEP) ? sweep_served : pop_count % total_lists;
        OpenList &list = open_lists[i][served];
        if (list.empty())
            continue;

        // Drain ineligible entries (stale, dead-end, already closed) from the
        // top of the served list at layer i until an expandable node surfaces
        // or the list is exhausted.
        OpenEntry current{StateID::no_state, 0, 0};
        bool found_expandable = false;
        while (!list.empty()) {
            const OpenEntry &candidate = list.top();
            SearchNode candidate_node =
                search_space.get_node(state_registry.lookup_state(candidate.id));
            if (candidate.g > candidate_node.get_g() ||
                candidate_node.is_dead_end() || candidate_node.is_closed()) {
                list.pop();
                if (layer_empty(i) && i == max_active_layer)
                    recompute_max_active_layer();
                continue;
            }
            current = candidate;
            list.pop();
            if (layer_empty(i) && i == max_active_layer)
                recompute_max_active_layer();
            found_expandable = true;
            break;
        }
        if (!found_expandable)
            continue;

        State state = state_registry.lookup_state(current.id);
        SearchNode node = search_space.get_node(state);

        node.close();
        statistics.inc_expanded();
        // POP schedule: advance the round-robin per expansion so the next
        // expansion (the next layer of this dive, or the next step) is guided
        // by the next heuristic. No-op for SWEEP.
        ++pop_count;

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(state, applicable_ops);
        pruning_method->prune_operators(state, applicable_ops);

        for (OperatorID op_id : applicable_ops) {
            OperatorProxy op = task_proxy.get_operators()[op_id];
            // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for NORMAL cost type.
            if (node.get_g() + op.get_cost() >= bound)
                continue;

            State succ_state = state_registry.get_successor_state(state, op);
            statistics.inc_generated();

            for (Evaluator *evaluator : path_dependent_evaluators) {
                evaluator->notify_state_transition(state, op_id, succ_state);
            }

            SearchNode succ_node = search_space.get_node(succ_state);
            if (succ_node.is_dead_end())
                continue;

            if (!reopen_closed_nodes && !succ_node.is_new())
                continue;

            int succ_g = node.get_g() + get_adjusted_cost(op);
            vector<int> succ_h;
            int prune_h = EvaluationResult::INFTY;

            // Compute the admissible h when it is needed: for the f-prune once
            // a bound exists, or unconditionally when it also feeds the pruner
            // queue (so the queue is populated even before the first
            // incumbent). The single computed value serves both.
            if (pruning_heuristic &&
                (use_pruner_queue || bound != numeric_limits<int>::max())) {
                EvaluationContext prune_ctx(
                    succ_state, succ_g, false, &statistics);
                prune_h = prune_ctx.get_evaluator_value_or_infinity(
                    pruning_heuristic.get());
                if (prune_h == EvaluationResult::INFTY) {
                    if (pruning_heuristic->dead_ends_are_reliable()) {
                        succ_node.mark_as_dead_end();
                        statistics.inc_dead_ends();
                    }
                    continue;
                }
                if (bound != numeric_limits<int>::max() && succ_g + prune_h >= bound)
                    continue;
            }

            if (succ_node.is_new()) {
                succ_node.open_new_node(node, op, get_adjusted_cost(op));
                if (!evaluate_and_prepare_node(
                        succ_state, succ_node, succ_g, succ_h, true))
                    continue;
            } else if (succ_node.is_closed() && reopen_closed_nodes) {
                if (succ_g >= succ_node.get_g())
                    continue;
                statistics.inc_reopened();
                succ_node.reopen_closed_node(node, op, get_adjusted_cost(op));
                if (!evaluate_and_prepare_node(
                        succ_state, succ_node, succ_g, succ_h, false))
                    continue;
            } else {
                if (succ_g < succ_node.get_g())
                    succ_node.update_open_node_parent(node, op, get_adjusted_cost(op));
                if (!evaluate_and_prepare_node(
                        succ_state, succ_node, succ_node.get_g(), succ_h, false))
                    continue;
            }

            if (task_properties::is_goal_state(task_proxy, succ_state)) {
                update_incumbent(succ_state);
                if (!anytime_search)
                    return SOLVED;
                continue;
            }

            // The guidance heuristics filled succ_h (size num_lists); append
            // the admissible value so the pruner queue is ranked too.
            if (use_pruner_queue)
                succ_h.push_back(prune_h);
            insert_successor(i + 1, succ_state.get_id(), succ_node.get_g(), succ_h);
        }
    }

    ++step_count;
    return IN_PROGRESS;
}

}
