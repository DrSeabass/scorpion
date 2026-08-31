#include "round_robin_triangle_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../evaluator.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"

#include "../utils/logging.h"

#include <algorithm>
#include <cassert>
#include <set>

using namespace std;

namespace round_robin_triangle_search {

RoundRobinTriangleSearch::RoundRobinTriangleSearch(
    const vector<shared_ptr<Evaluator>> &evals,
    int slope,
    bool reopen_closed,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      slope(slope),
      reopen_closed_nodes(reopen_closed),
      evals(evals),
      num_lists(static_cast<int>(evals.size())) {
    if (slope <= 0) {
        cerr << "RoundRobinTriangleSearch: slope must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (evals.empty()) {
        cerr << "RoundRobinTriangleSearch: at least one evaluator is required."
             << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void RoundRobinTriangleSearch::initialize() {
    log << "Conducting eager per-depth round-robin triangle search with slope "
        << slope << ", " << num_lists << " heuristic(s), (real) bound = "
        << bound << endl;

    set<Evaluator *> path_dependent;
    for (const shared_ptr<Evaluator> &eval : evals)
        eval->get_path_dependent_evaluators(path_dependent);
    path_dependent_evaluators.assign(path_dependent.begin(), path_dependent.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators)
        evaluator->notify_initial_state(initial_state);

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    bool is_dead_end = false;
    vector<int> initial_h;
    initial_h.reserve(num_lists);
    for (const shared_ptr<Evaluator> &eval : evals) {
        int h = eval_context.get_evaluator_value_or_infinity(eval.get());
        if (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable())
            is_dead_end = true;
        initial_h.push_back(h);
    }

    extend_layers(1);
    if (is_dead_end) {
        log << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context))
            statistics.print_checkpoint_line(0);
        start_evaluator_statistics(eval_context);

        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();
        if (task_properties::is_goal_state(task_proxy, initial_state)) {
            set_plan({});
        } else {
            insert_successor(0, initial_state.get_id(), 0, initial_h);
        }
    }
    print_initial_evaluator_values(eval_context);
}

void RoundRobinTriangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

void RoundRobinTriangleSearch::start_evaluator_statistics(
    EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(evals[0].get());
    if (value != EvaluationResult::INFTY)
        statistics.report_f_value_progress(value);
}

void RoundRobinTriangleSearch::extend_layers(int num_layers) {
    for (int i = 0; i < num_layers; ++i)
        layers.emplace_back(num_lists);
}

bool RoundRobinTriangleSearch::layer_empty(int layer) const {
    for (const OpenList &list : layers[layer].lists) {
        if (!list.empty())
            return false;
    }
    return true;
}

void RoundRobinTriangleSearch::recompute_max_active_layer() {
    while (max_active_layer >= 0 && layer_empty(max_active_layer))
        --max_active_layer;
}

bool RoundRobinTriangleSearch::evaluate_and_prepare_node(
    const State &state, SearchNode &node, int g,
    vector<int> &h_out, bool is_new_evaluation) {
    EvaluationContext eval_context(state, g, false, &statistics);
    if (is_new_evaluation)
        statistics.inc_evaluated_states();

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
    if (is_new_evaluation && search_progress.check_progress(eval_context))
        statistics.print_checkpoint_line(node.get_g());
    return true;
}

void RoundRobinTriangleSearch::insert_successor(
    int layer, StateID id, int g, const vector<int> &hs) {
    assert(layer >= 0);
    assert(static_cast<int>(hs.size()) == num_lists);
    if (layer >= static_cast<int>(layers.size()))
        extend_layers(layer + 1 - static_cast<int>(layers.size()));
    for (int k = 0; k < num_lists; ++k)
        layers[layer].lists[k].push({id, hs[k], g});
    max_active_layer = max(max_active_layer, layer);
}

SearchStatus RoundRobinTriangleSearch::step() {
    while (!layers.empty() && layer_empty(0)) {
        layers.pop_front();
        --max_active_layer;
    }
    max_active_layer = max(max_active_layer, 0);

    if (layers.empty()) {
        if (found_solution())
            return SOLVED;
        log << "All open lists are empty -- no solution!" << endl;
        return FAILED;
    }

    const int cascade_cap = max_active_layer + slope;
    for (int i = 0; i < cascade_cap; ++i) {
        if (i >= static_cast<int>(layers.size()))
            break;

        Layer &layer = layers[i];
        const int served = layer.next_served;
        OpenList &list = layer.lists[served];
        if (list.empty())
            continue;

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

        // Rotate this depth only after it has actually supplied a live node.
        layer.next_served = (served + 1) % num_lists;

        State state = state_registry.lookup_state(current.id);
        SearchNode node = search_space.get_node(state);
        node.close();
        statistics.inc_expanded();

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(state, applicable_ops);
        for (OperatorID op_id : applicable_ops) {
            OperatorProxy op = task_proxy.get_operators()[op_id];
            if (node.get_g() + op.get_cost() >= bound)
                continue;

            State succ_state = state_registry.get_successor_state(state, op);
            statistics.inc_generated();
            for (Evaluator *evaluator : path_dependent_evaluators)
                evaluator->notify_state_transition(state, op_id, succ_state);

            SearchNode succ_node = search_space.get_node(succ_state);
            if (succ_node.is_dead_end())
                continue;
            if (!reopen_closed_nodes && !succ_node.is_new())
                continue;

            int succ_g = node.get_g() + get_adjusted_cost(op);
            vector<int> succ_h;
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
                    succ_node.update_open_node_parent(
                        node, op, get_adjusted_cost(op));
                if (!evaluate_and_prepare_node(
                        succ_state, succ_node, succ_node.get_g(), succ_h, false))
                    continue;
            }

            if (task_properties::is_goal_state(task_proxy, succ_state)) {
                Plan plan = search_space.trace_path(
                    task_proxy, successor_generator, succ_state);
                set_plan(plan);
                return SOLVED;
            }
            insert_successor(
                i + 1, succ_state.get_id(), succ_node.get_g(), succ_h);
        }
    }
    return IN_PROGRESS;
}

}
