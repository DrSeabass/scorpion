#include "lazy_triangle_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../plan_manager.h"
#include "../pruning_method.h"

#include "../open_lists/best_first_open_list.h"
#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"
#include "../utils/system.h"

#include <cassert>
#include <set>

using namespace std;

namespace lazy_triangle_search {

LazyTriangleSearch::LazyTriangleSearch(
    const shared_ptr<Evaluator> &eval, int slope, bool reopen_closed,
    bool anytime, const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      slope(slope),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      eval(eval),
      pruning_method(pruning),
      root_pending(true) {
    if (slope <= 0) {
        cerr << "LazyTriangleSearch: slope must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    open_list_factory =
        make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(
            eval, false);
}

void LazyTriangleSearch::initialize() {
    log << "Conducting lazy triangle search with slope " << slope
        << ", (real) bound = " << bound << endl;

    assert(eval);
    assert(open_list_factory);

    set<Evaluator *> evals;
    eval->get_path_dependent_evaluators(evals);
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    open_lists.clear();
    open_lists.push_back(create_open_list());
    root_pending = true;

    pruning_method->initialize(task);
}

void LazyTriangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

void LazyTriangleSearch::start_evaluator_statistics(
    EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

bool LazyTriangleSearch::has_non_empty_lists() const {
    if (root_pending)
        return true;
    for (const unique_ptr<EdgeOpenList> &open_list : open_lists) {
        if (!open_list->empty())
            return true;
    }
    return false;
}

unique_ptr<EdgeOpenList> LazyTriangleSearch::create_open_list() const {
    return open_list_factory->create_edge_open_list();
}

void LazyTriangleSearch::extend_open_lists(int num_lists) {
    for (int i = 0; i < num_lists; ++i) {
        open_lists.push_back(create_open_list());
    }
}

void LazyTriangleSearch::trim_empty_lists() {
    while (!open_lists.empty() && open_lists.front()->empty()) {
        open_lists.pop_front();
    }
    while (!open_lists.empty() && open_lists.back()->empty()) {
        open_lists.pop_back();
    }
}

void LazyTriangleSearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        log << "LazyTriangleSearch: improved incumbent with cost "
            << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

LazyTriangleSearch::ExpansionOutcome LazyTriangleSearch::process_candidate(
    const State &state, StateID predecessor_id, OperatorID operator_id, int g,
    int real_g, int source_list_index, bool is_root) {
    SearchNode node = search_space.get_node(state);

    if (!reopen_closed_nodes && !node.is_new())
        return ExpansionOutcome::SKIPPED;

    bool reopen = reopen_closed_nodes && node.is_closed() &&
                  !node.is_dead_end() && (g < node.get_g());

    if (!(node.is_new() || reopen))
        return ExpansionOutcome::SKIPPED;

    if (!is_root) {
        assert(predecessor_id != StateID::no_state);
        assert(operator_id != OperatorID::no_operator);
        if (!path_dependent_evaluators.empty()) {
            State parent_state = state_registry.lookup_state(predecessor_id);
            for (Evaluator *evaluator : path_dependent_evaluators) {
                evaluator->notify_state_transition(
                    parent_state, operator_id, state);
            }
        }
    }

    statistics.inc_evaluated_states();
    EvaluationContext eval_context(state, g, true, &statistics);

    int h = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
        node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return ExpansionOutcome::SKIPPED;
    }

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
        return anytime_search ? ExpansionOutcome::SKIPPED
                              : ExpansionOutcome::SOLVED;
    }

    node.close();
    statistics.inc_expanded();

    if (search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(g);
    }

    while (static_cast<int>(open_lists.size()) <= source_list_index + 1) {
        open_lists.push_back(create_open_list());
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
        EvaluationContext succ_eval_context(
            eval_context, succ_g, false, nullptr);
        open_lists[source_list_index + 1]->insert(
            succ_eval_context, make_pair(state.get_id(), op_id));
    }

    return ExpansionOutcome::EXPANDED;
}

SearchStatus LazyTriangleSearch::step() {
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

    extend_open_lists(slope);

    const int num_lists = static_cast<int>(open_lists.size());
    for (int i = 0; i < num_lists - 1; ++i) {
        if (i == 0 && root_pending) {
            root_pending = false;
            State initial_state = state_registry.get_initial_state();
            ExpansionOutcome outcome = process_candidate(
                initial_state, StateID::no_state, OperatorID::no_operator, 0, 0,
                i, true);
            if (outcome == ExpansionOutcome::SOLVED)
                return SOLVED;
            continue;
        }

        bool expanded = false;
        while (!open_lists[i]->empty() && !expanded) {
            EdgeOpenListEntry next = open_lists[i]->remove_min();

            StateID predecessor_id = next.first;
            OperatorID operator_id = next.second;
            if (predecessor_id == StateID::no_state ||
                operator_id == OperatorID::no_operator)
                continue;

            State predecessor = state_registry.lookup_state(predecessor_id);
            OperatorProxy op = task_proxy.get_operators()[operator_id];
            if (!task_properties::is_applicable(op, predecessor))
                continue;

            SearchNode predecessor_node = search_space.get_node(predecessor);
            int g = predecessor_node.get_g() + get_adjusted_cost(op);
            // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for
            // NORMAL cost type.
            int real_g = predecessor_node.get_g() + op.get_cost();
            State state = state_registry.get_successor_state(predecessor, op);

            ExpansionOutcome outcome = process_candidate(
                state, predecessor_id, operator_id, g, real_g, i, false);
            if (outcome == ExpansionOutcome::SOLVED)
                return SOLVED;
            expanded = (outcome == ExpansionOutcome::EXPANDED);
        }
    }

    trim_empty_lists();
    return IN_PROGRESS;
}

}
