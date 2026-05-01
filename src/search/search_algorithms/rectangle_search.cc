#include "rectangle_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../evaluator.h"
#include "../pruning_method.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"

#include "../utils/logging.h"

#include <algorithm>
#include <cassert>

using namespace std;

namespace rectangle_search {

RectangleSearch::RectangleSearch(
    const shared_ptr<Evaluator> &eval,
    int beam_width,
    int aspect,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      beam_width(beam_width),
      aspect(aspect),
      eval(eval),
      pruning_method(pruning),
      depth(1) {
    if (beam_width <= 0) {
        cerr << "RectangleSearch: beam_width must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (aspect <= 0) {
        cerr << "RectangleSearch: aspect must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void RectangleSearch::initialize() {
    log << "Conducting rectangle search with beam width " << beam_width
        << ", aspect = " << aspect
        << ", (real) bound = " << bound << endl;

    assert(eval);

    set<Evaluator *> evals;
    eval->get_path_dependent_evaluators(evals);
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

    if (is_dead_end) {
        log << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context)) {
            statistics.print_checkpoint_line(0);
        }
        start_evaluator_statistics(eval_context);

        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();

        open_lists.clear();
        open_lists.push_back(deque<StateID>());

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(initial_state, applicable_ops);
        pruning_method->prune_operators(initial_state, applicable_ops);

        node.close();
        statistics.inc_expanded();

        for (OperatorID op_id : applicable_ops) {
            OperatorProxy op = task_proxy.get_operators()[op_id];
            if (op.get_cost() >= bound)
                continue;

            State succ_state = state_registry.get_successor_state(initial_state, op);
            statistics.inc_generated();

            for (Evaluator *evaluator : path_dependent_evaluators) {
                evaluator->notify_state_transition(initial_state, op_id, succ_state);
            }

            SearchNode succ_node = search_space.get_node(succ_state);
            if (succ_node.is_dead_end())
                continue;

            int succ_g = get_adjusted_cost(op);

            if (succ_node.is_new()) {
                EvaluationContext succ_eval_context(succ_state, succ_g, false, &statistics);
                statistics.inc_evaluated_states();

                int eval_h = succ_eval_context.get_evaluator_value_or_infinity(eval.get());
                if (eval_h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
                    succ_node.mark_as_dead_end();
                    statistics.inc_dead_ends();
                    continue;
                }

                succ_node.open_new_node(node, op, get_adjusted_cost(op));

                if (search_progress.check_progress(succ_eval_context)) {
                    statistics.print_checkpoint_line(succ_node.get_g());
                }

                insert_into_open_list(0, {succ_state.get_id(), eval_h, succ_g});

                if (check_goal_and_set_plan(succ_state))
                    break;
            }
        }

        depth = 1;

        if (open_lists.size() == 1) {
            open_lists.push_back(deque<StateID>());
        }
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void RectangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

void RectangleSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

bool RectangleSearch::select_and_expand(int list_index) {
    if (list_index >= static_cast<int>(open_lists.size()) || open_lists[list_index].empty())
        return false;

    StateID state_id = open_lists[list_index].front();
    open_lists[list_index].pop_front();

    State state = state_registry.lookup_state(state_id);
    SearchNode node = search_space.get_node(state);

    if (check_goal_and_set_plan(state))
        return true;

    if (node.is_closed())
        return false;

    node.close();
    statistics.inc_expanded();

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

        SearchNode succ_node = search_space.get_node(succ_state);

        for (Evaluator *evaluator : path_dependent_evaluators) {
            evaluator->notify_state_transition(state, op_id, succ_state);
        }

        if (succ_node.is_dead_end() || succ_node.is_closed())
            continue;

        int succ_g = node.get_g() + get_adjusted_cost(op);

        if (succ_node.is_new()) {
            EvaluationContext succ_eval_context(succ_state, succ_g, false, &statistics);
            statistics.inc_evaluated_states();

            int eval_h = succ_eval_context.get_evaluator_value_or_infinity(eval.get());
            if (eval_h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
                succ_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                continue;
            }

            succ_node.open_new_node(node, op, get_adjusted_cost(op));

            if (search_progress.check_progress(succ_eval_context)) {
                statistics.print_checkpoint_line(succ_node.get_g());
            }

            while (static_cast<int>(open_lists.size()) <= list_index + 1) {
                open_lists.push_back(deque<StateID>());
            }

            insert_into_open_list(list_index + 1, {succ_state.get_id(), eval_h, succ_g});

            if (check_goal_and_set_plan(succ_state))
                return true;
        } else {
            if (succ_node.get_g() > succ_g) {
                succ_node.update_open_node_parent(node, op, get_adjusted_cost(op));
            }

            EvaluationContext succ_eval_context(succ_state, succ_node.get_g(), false, &statistics);
            statistics.inc_evaluated_states();

            int eval_h = succ_eval_context.get_evaluator_value_or_infinity(eval.get());
            if (eval_h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
                succ_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                continue;
            }

            while (static_cast<int>(open_lists.size()) <= list_index + 1) {
                open_lists.push_back(deque<StateID>());
            }

            insert_into_open_list(list_index + 1, {succ_state.get_id(), eval_h, succ_node.get_g()});

            if (check_goal_and_set_plan(succ_state))
                return true;
        }
    }

    return false;
}

void RectangleSearch::insert_into_open_list(int list_index, const Candidate &candidate) {
    assert(list_index >= 0 && list_index < static_cast<int>(open_lists.size()));

    deque<StateID> &open_list = open_lists[list_index];

    for (StateID id : open_list) {
        if (id == candidate.id)
            return;
    }

    auto it = open_list.begin();
    for (; it != open_list.end(); ++it) {
        State existing_state = state_registry.lookup_state(*it);
        SearchNode existing_node = search_space.get_node(existing_state);

        EvaluationContext existing_eval_context(
            existing_state, existing_node.get_g(), false, &statistics);
        int existing_value = existing_eval_context.get_evaluator_value_or_infinity(eval.get());

        if (candidate.eval_value <= existing_value)
            break;
    }

    open_list.insert(it, candidate.id);
}

void RectangleSearch::extend_open_lists(int num_lists) {
    for (int i = 0; i < num_lists; ++i) {
        open_lists.push_back(deque<StateID>());
    }
}

void RectangleSearch::trim_empty_lists() {
    while (!open_lists.empty() && open_lists.front().empty()) {
        open_lists.erase(open_lists.begin());
    }
    while (!open_lists.empty() && open_lists.back().empty()) {
        open_lists.pop_back();
    }
}

bool RectangleSearch::has_non_empty_lists() const {
    for (const auto &open_list : open_lists) {
        if (!open_list.empty())
            return true;
    }
    return false;
}

SearchStatus RectangleSearch::step() {
    if (found_solution())
        return SOLVED;

    if (!has_non_empty_lists()) {
        log << "All open lists are empty -- no solution!" << endl;
        return FAILED;
    }

    if (open_lists.size() == 1) {
        open_lists.push_back(deque<StateID>());
    }

    const int initial_num_lists = static_cast<int>(open_lists.size());

    for (int i = 0; i < initial_num_lists - 1; ++i) {
        for (int w = 0; w < beam_width; ++w) {
            if (open_lists[i].empty())
                break;
            if (select_and_expand(i))
                return SOLVED;
        }

        extend_open_lists(aspect);

        const int current_num_lists = static_cast<int>(open_lists.size());
        for (int j = i + 1; j < current_num_lists - 1; ++j) {
            for (int k = 0; k < depth; ++k) {
                for (int w = 0; w < beam_width; ++w) {
                    if (open_lists[j].empty())
                        break;
                    if (select_and_expand(j))
                        return SOLVED;
                }
            }
        }
    }

    depth += aspect;
    trim_empty_lists();
    return IN_PROGRESS;
}

}
