#include "adaptive_triangle_search.h"

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

using namespace std;

namespace adaptive_triangle_search {

AdaptiveTriangleSearch::AdaptiveTriangleSearch(
    const shared_ptr<Evaluator> &eval,
    bool reopen_closed,
    bool anytime,
    bool lift_floor,
    const shared_ptr<Evaluator> &pruning_heuristic,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      lift_floor(lift_floor),
      eval(eval),
      pruning_heuristic(pruning_heuristic),
      pruning_method(pruning) {
}

void AdaptiveTriangleSearch::initialize() {
    log << "Conducting adaptive triangle search, lift_floor = " << lift_floor
        << ", (real) bound = " << bound << endl;

    assert(eval);

    set<Evaluator *> evals;
    eval->get_path_dependent_evaluators(evals);
    if (pruning_heuristic) {
        pruning_heuristic->get_path_dependent_evaluators(evals);
    }
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    bool is_dead_end = false;
    int h = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
        is_dead_end = true;
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
            insert_into_open_list(0, {initial_state.get_id(), h, 0});
        }
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void AdaptiveTriangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

void AdaptiveTriangleSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

void AdaptiveTriangleSearch::extend_open_lists(int num_lists) {
    for (int i = 0; i < num_lists; ++i) {
        open_lists.emplace_back();
    }
}

void AdaptiveTriangleSearch::recompute_max_active_layer() {
    while (max_active_layer >= 0 && open_lists[max_active_layer].empty()) {
        --max_active_layer;
    }
}

void AdaptiveTriangleSearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        log << "AdaptiveTriangleSearch: improved incumbent with cost " << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

bool AdaptiveTriangleSearch::evaluate_and_prepare_node(
    const State &state, SearchNode &node, int g, int &h_out,
    bool is_new_evaluation) {
    EvaluationContext eval_context(state, g, false, &statistics);
    if (is_new_evaluation)
        statistics.inc_evaluated_states();

    int h = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
        node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return false;
    }

    if (is_new_evaluation && search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(node.get_g());
    }

    h_out = h;
    return true;
}

void AdaptiveTriangleSearch::insert_into_open_list(int list_index, const OpenEntry &entry) {
    assert(list_index >= 0 && list_index < static_cast<int>(open_lists.size()));
    open_lists[list_index].push(entry);
    if (list_index > max_active_layer)
        max_active_layer = list_index;
}

SearchStatus AdaptiveTriangleSearch::step() {
    while (!open_lists.empty() && open_lists.front().empty()) {
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

    // Per-step heuristic-trend budget. Budget pays one unit to instantiate a
    // new deque slot at the frontier; informed layer-transitions (h(curr) <
    // h(prev_expanded)) refund one unit, uninformed transitions debit one.
    // last_expanded_h tracks the h of the most recently expanded node in this
    // step (preserved across skipped empty layers).
    int budget = 1;
    int last_expanded_h = 0;
    bool have_last_h = false;

    // Direction B (relaxed cascade start-depth). With lift_floor, begin the
    // cascade prev_layers_added-1 layers below the shallowest active layer
    // (which the front-drain above pins to index 0) instead of at the root,
    // skipping the shallow layers the previous step's realized dive said we
    // trust. prev_layers_added is the emergent analog of ratchet's persistent
    // slope: the number of new frontier layers the last step instantiated. The
    // clamp keeps the deepest active layer served (>=1 expansion/step, floor
    // can't run off the deque). The floor self-resets toward the root because
    // an unproductive (e.g. tail-pruned) step instantiates few/no new layers,
    // shrinking prev_layers_added. lift_floor off yields start 0 == vanilla
    // adaptive. layers_added counts this step's frontier instantiations to
    // carry forward as the next step's floor.
    const int cascade_start =
        lift_floor ? max(0, min(prev_layers_added - 1, max_active_layer)) : 0;
    int layers_added = 0;

    for (int i = cascade_start; ; ++i) {
        if (i >= static_cast<int>(open_lists.size()))
            break;

        if (open_lists[i].empty())
            continue;

        // Drain ineligible entries (stale, dead-end, already closed) from
        // the top of layer i without committing to expansion yet -- we don't
        // want to pay the frontier-extension cost until we know we have
        // someone to expand here. The expandable entry is identified but
        // left on top until the cost check passes.
        OpenEntry current{StateID::no_state, 0, 0};
        bool found_expandable = false;
        while (!open_lists[i].empty()) {
            const OpenEntry &candidate = open_lists[i].top();
            SearchNode candidate_node =
                search_space.get_node(state_registry.lookup_state(candidate.id));
            if (candidate.g > candidate_node.get_g() ||
                candidate_node.is_dead_end() || candidate_node.is_closed()) {
                open_lists[i].pop();
                if (open_lists[i].empty() && i == max_active_layer)
                    recompute_max_active_layer();
                continue;
            }
            current = candidate;
            found_expandable = true;
            break;
        }
        if (!found_expandable)
            continue;

        // We have an expandable top. Ensure layer i+1 exists to receive its
        // successors; free if already in the deque, otherwise pay one budget
        // unit. If we can't afford it, halt the cascade with the expandable
        // entry still in place for the next step.
        if (i + 1 >= static_cast<int>(open_lists.size())) {
            if (budget <= 0)
                break;
            --budget;
            extend_open_lists(1);
            ++layers_added;
        }

        // Commit: actually pop the expandable entry.
        open_lists[i].pop();
        if (open_lists[i].empty() && i == max_active_layer)
            recompute_max_active_layer();

        State state = state_registry.lookup_state(current.id);
        SearchNode node = search_space.get_node(state);

        node.close();
        statistics.inc_expanded();

        if (have_last_h) {
            if (current.h < last_expanded_h)
                ++budget;
            else
                --budget;
        }
        last_expanded_h = current.h;
        have_last_h = true;

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
            int succ_h = EvaluationResult::INFTY;

            if (pruning_heuristic &&
                bound != numeric_limits<int>::max()) {
                EvaluationContext prune_ctx(
                    succ_state, succ_g, false, &statistics);
                int prune_h = prune_ctx.get_evaluator_value_or_infinity(
                    pruning_heuristic.get());
                if (prune_h == EvaluationResult::INFTY) {
                    if (pruning_heuristic->dead_ends_are_reliable()) {
                        succ_node.mark_as_dead_end();
                        statistics.inc_dead_ends();
                    }
                    continue;
                }
                if (succ_g + prune_h >= bound)
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

            insert_into_open_list(i + 1, {succ_state.get_id(), succ_h, succ_node.get_g()});
        }
    }

    // Carry this step's realized dive depth to drive next step's lift_floor.
    prev_layers_added = layers_added;
    return IN_PROGRESS;
}

}
