#include "anytime_nonparametric_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../evaluator.h"
#include "../plan_manager.h"
#include "../pruning_method.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"

#include "../utils/logging.h"

#include <cassert>
#include <limits>
#include <vector>

using namespace std;

namespace anytime_nonparametric_search {

AnytimeNonparametricSearch::AnytimeNonparametricSearch(
    const shared_ptr<Evaluator> &eval,
    bool reopen_closed,
    bool anytime,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      eval(eval),
      pruning_method(pruning),
      // The comparator reads the incumbent bound G through &bound; the base
      // class has already initialized bound by the time this member runs.
      open_list(OpenEntryCompare{&bound}) {
}

void AnytimeNonparametricSearch::initialize() {
    log << "Conducting anytime nonparametric (ANA*) search"
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
        if (task_properties::is_goal_state(task_proxy, initial_state)) {
            bool improved = false;
            update_incumbent(initial_state, improved);
        } else {
            open_list.push({initial_state.get_id(), 0, h});
        }
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void AnytimeNonparametricSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
    if (found_solution() && suboptimality_bound != numeric_limits<double>::infinity()) {
        log << "ANA*: proven suboptimality bound " << suboptimality_bound << endl;
    }
}

void AnytimeNonparametricSearch::start_evaluator_statistics(
    EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

void AnytimeNonparametricSearch::update_incumbent(
    const State &goal_state, bool &improved) {
    improved = false;
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        improved = true;
        log << "ANA*: improved incumbent with cost " << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

// Re-key the open list after the incumbent bound G has changed. Each entry's
// potential e(s) = (G - g)/h shifts with G, so the heap must be rebuilt;
// while rebuilding we drop entries that became stale, dead, closed, or that
// the tightened bound now prunes (g + h >= G, i.e. e(s) <= 1).
void AnytimeNonparametricSearch::reorder_open() {
    vector<OpenEntry> kept;
    kept.reserve(open_list.size());
    while (!open_list.empty()) {
        OpenEntry entry = open_list.top();
        open_list.pop();

        State state = state_registry.lookup_state(entry.id);
        SearchNode node = search_space.get_node(state);
        if (entry.g > node.get_g() || node.is_dead_end() || node.is_closed())
            continue;
        if (static_cast<long long>(entry.g) + entry.h >= bound)
            continue;
        kept.push_back(entry);
    }
    open_list = OpenList(OpenEntryCompare{&bound}, move(kept));
}

bool AnytimeNonparametricSearch::evaluate_and_prepare_node(
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

SearchStatus AnytimeNonparametricSearch::step() {
    // Pop the highest-potential eligible node, draining stale entries (a
    // cheaper path was found, or the node is closed/dead) and any that the
    // current bound prunes.
    OpenEntry current{StateID::no_state, 0, 0};
    bool found_expandable = false;
    while (!open_list.empty()) {
        OpenEntry candidate = open_list.top();
        open_list.pop();

        State state = state_registry.lookup_state(candidate.id);
        SearchNode node = search_space.get_node(state);
        if (candidate.g > node.get_g() || node.is_dead_end() ||
            node.is_closed())
            continue;
        if (static_cast<long long>(candidate.g) + candidate.h >= bound)
            continue;
        current = candidate;
        found_expandable = true;
        break;
    }

    if (!found_expandable) {
        if (found_solution()) {
            // Every remaining node was pruned by g + h >= G, so no node can
            // beat the incumbent: it is optimal (suboptimality bound 1.0).
            suboptimality_bound = 1.0;
            log << "Open list is empty -- incumbent proven optimal "
                << "(suboptimality bound 1.0)." << endl;
            return SOLVED;
        }
        log << "Open list is empty -- no solution!" << endl;
        return FAILED;
    }

    State state = state_registry.lookup_state(current.id);
    SearchNode node = search_space.get_node(state);

    // Once an incumbent exists, the potential of the node we are about to
    // expand (the open maximum) is the current suboptimality bound on it.
    if (found_solution() && current.h > 0) {
        double e = static_cast<double>(bound - current.g) / current.h;
        if (e < suboptimality_bound) {
            suboptimality_bound = e;
            log << "ANA*: suboptimality bound tightened to "
                << suboptimality_bound << endl;
        }
    }

    node.close();
    statistics.inc_expanded();

    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(state, applicable_ops);
    pruning_method->prune_operators(state, applicable_ops);

    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for
        // NORMAL cost type.
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

        if (succ_node.is_new()) {
            succ_node.open_new_node(node, op, get_adjusted_cost(op));
            if (!evaluate_and_prepare_node(succ_state, succ_node, succ_g, succ_h, true))
                continue;
        } else if (succ_node.is_closed() && reopen_closed_nodes) {
            if (succ_g >= succ_node.get_g())
                continue;
            statistics.inc_reopened();
            succ_node.reopen_closed_node(node, op, get_adjusted_cost(op));
            if (!evaluate_and_prepare_node(succ_state, succ_node, succ_g, succ_h, false))
                continue;
        } else {
            if (succ_g < succ_node.get_g())
                succ_node.update_open_node_parent(node, op, get_adjusted_cost(op));
            if (!evaluate_and_prepare_node(
                    succ_state, succ_node, succ_node.get_g(), succ_h, false))
                continue;
        }

        if (task_properties::is_goal_state(task_proxy, succ_state)) {
            bool improved = false;
            update_incumbent(succ_state, improved);
            if (!anytime_search)
                return SOLVED;
            // G changed: re-key the open list before inserting anything else.
            if (improved)
                reorder_open();
            continue;
        }

        // ANA* line 13: keep s only if it can still improve the incumbent.
        int succ_g_now = succ_node.get_g();
        if (static_cast<long long>(succ_g_now) + succ_h >= bound)
            continue;

        open_list.push({succ_state.get_id(), succ_g_now, succ_h});
    }

    return IN_PROGRESS;
}

}
