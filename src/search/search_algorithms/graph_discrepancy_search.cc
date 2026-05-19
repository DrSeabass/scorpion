#include "graph_discrepancy_search.h"

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
#include <numeric>

using namespace std;

namespace graph_discrepancy_search {

GraphDiscrepancySearch::GraphDiscrepancySearch(
    const shared_ptr<Evaluator> &eval,
    const shared_ptr<Evaluator> &prune_eval_arg, bool reopen_closed,
    bool anytime, const string &discrepancy_mode_str,
    const shared_ptr<PruningMethod> &pruning, OperatorCost cost_type, int bound,
    double max_time, const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      discrepancy_mode([&discrepancy_mode_str]() {
          if (discrepancy_mode_str == "binary")
              return DiscrepancyMode::BINARY;
          if (discrepancy_mode_str == "child_rank")
              return DiscrepancyMode::CHILD_RANK;
          if (discrepancy_mode_str == "h_gap")
              return DiscrepancyMode::H_GAP;
          cerr << "GraphDiscrepancySearch: unsupported discrepancy_mode '"
               << discrepancy_mode_str
               << "'. Expected one of: binary, child_rank, h_gap." << endl;
          utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
      }()),
      discrepancy_eval(eval),
      prune_eval(prune_eval_arg ? prune_eval_arg : eval),
      pruning_method(pruning),
      state_discrepancy(numeric_limits<int>::max()),
      next_insertion_id(0),
      has_incumbent(false),
      incumbent_cost(numeric_limits<int>::max()) {
    if (!discrepancy_eval) {
        cerr
            << "GraphDiscrepancySearch: an evaluator must be provided via option 'eval'."
            << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (!prune_eval_arg) {
        log << "GDS: no prune_eval provided, using eval for incumbent pruning."
            << endl;
    }
}

void GraphDiscrepancySearch::initialize() {
    log << "Conducting graph discrepancy search"
        << (anytime_search ? " (anytime)" : " (satisficing)")
        << (reopen_closed_nodes ? " with" : " without")
        << " reopening closed nodes, discrepancy_mode = "
        << (discrepancy_mode == DiscrepancyMode::BINARY
                ? "binary"
                : (discrepancy_mode == DiscrepancyMode::CHILD_RANK
                       ? "child_rank"
                       : "h_gap"))
        << ", (real) bound = " << bound << endl;

    set<Evaluator *> evals;
    discrepancy_eval->get_path_dependent_evaluators(evals);
    prune_eval->get_path_dependent_evaluators(evals);
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    bool dead_end = false;
    int rank_h =
        eval_context.get_evaluator_value_or_infinity(discrepancy_eval.get());
    if (rank_h == EvaluationResult::INFTY &&
        discrepancy_eval->dead_ends_are_reliable())
        dead_end = true;
    int prune_h =
        eval_context.get_evaluator_value_or_infinity(prune_eval.get());
    if (prune_h == EvaluationResult::INFTY &&
        prune_eval->dead_ends_are_reliable())
        dead_end = true;

    if (dead_end) {
        log << "Initial state is a dead end." << endl;
        SearchNode initial_node = search_space.get_node(initial_state);
        initial_node.mark_as_dead_end();
        statistics.inc_dead_ends();
    } else {
        if (search_progress.check_progress(eval_context)) {
            statistics.print_checkpoint_line(0);
        }
        start_evaluator_statistics(eval_context);

        SearchNode initial_node = search_space.get_node(initial_state);
        initial_node.open_initial();
        state_discrepancy[initial_state] = 0;
        push_open(initial_state, 0, 0);
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void GraphDiscrepancySearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
    if (has_incumbent) {
        log << "Best solution cost: " << incumbent_cost << endl;
    }
}

void GraphDiscrepancySearch::start_evaluator_statistics(
    EvaluationContext &eval_context) {
    int value =
        eval_context.get_evaluator_value_or_infinity(discrepancy_eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

void GraphDiscrepancySearch::push_open(
    const State &state, int total_discrepancy, int g) {
    open.push({state.get_id(), total_discrepancy, g, next_insertion_id++});
}

bool GraphDiscrepancySearch::evaluate_successor(
    const SearchNode &parent_node, const OperatorProxy &op, const State &state,
    SearchNode &node, int succ_g, int &rank_h_out) {
    EvaluationContext eval_context(state, succ_g, false, &statistics);
    statistics.inc_evaluated_states();

    int rank_h =
        eval_context.get_evaluator_value_or_infinity(discrepancy_eval.get());
    if (rank_h == EvaluationResult::INFTY &&
        discrepancy_eval->dead_ends_are_reliable()) {
        node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return false;
    }

    int prune_h =
        eval_context.get_evaluator_value_or_infinity(prune_eval.get());
    if (prune_h == EvaluationResult::INFTY &&
        prune_eval->dead_ends_are_reliable()) {
        node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return false;
    }

    // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for NORMAL
    // cost type.
    if (has_incumbent &&
        parent_node.get_g() + op.get_cost() + prune_h >= incumbent_cost)
        return false;

    if (search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(succ_g);
    }

    rank_h_out = rank_h;
    return true;
}

void GraphDiscrepancySearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!has_incumbent || candidate_cost < incumbent_cost) {
        has_incumbent = true;
        incumbent_cost = candidate_cost;
        set_plan(candidate_plan);
        if (candidate_cost < bound)
            bound = candidate_cost;
        log << "GDS: improved incumbent with cost " << incumbent_cost << endl;
        plan_manager.save_plan(candidate_plan, task_proxy, true);
    }
}

SearchStatus GraphDiscrepancySearch::step() {
    while (!open.empty()) {
        OpenEntry entry = open.top();
        open.pop();

        State state = state_registry.lookup_state(entry.id);
        SearchNode node = search_space.get_node(state);

        if (node.is_dead_end() || node.is_closed())
            continue;

        int recorded_discrepancy = state_discrepancy[state];
        if (entry.g != node.get_g() ||
            entry.total_discrepancy != recorded_discrepancy)
            continue;

        if (task_properties::is_goal_state(task_proxy, state)) {
            update_incumbent(state);
            if (!anytime_search)
                return SOLVED;
            node.close();
            continue;
        }

        node.close();
        statistics.inc_expanded();

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(state, applicable_ops);
        pruning_method->prune_operators(state, applicable_ops);

        vector<SuccessorCandidate> candidates;
        candidates.reserve(applicable_ops.size());

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

            int succ_g = node.get_g() + get_adjusted_cost(op);

            if (!succ_node.is_new()) {
                if (succ_g >= succ_node.get_g())
                    continue;
            }

            int rank_h = EvaluationResult::INFTY;
            if (!evaluate_successor(
                    node, op, succ_state, succ_node, succ_g, rank_h))
                continue;

            candidates.push_back(
                {op_id, succ_state, succ_node, succ_g, rank_h});
        }

        if (candidates.empty())
            continue;

        vector<int> sorted_indices(candidates.size());
        iota(sorted_indices.begin(), sorted_indices.end(), 0);
        stable_sort(
            sorted_indices.begin(), sorted_indices.end(),
            [&candidates](int lhs, int rhs) {
                const SuccessorCandidate &a = candidates[lhs];
                const SuccessorCandidate &b = candidates[rhs];
                if (a.rank_h != b.rank_h)
                    return a.rank_h < b.rank_h;
                if (a.succ_g != b.succ_g)
                    return a.succ_g < b.succ_g;
                return a.op_id.get_index() < b.op_id.get_index();
            });

        vector<int> discrepancies(candidates.size(), 1);
        if (discrepancy_mode == DiscrepancyMode::CHILD_RANK) {
            for (size_t rank = 0; rank < sorted_indices.size(); ++rank) {
                discrepancies[sorted_indices[rank]] = static_cast<int>(rank);
            }
        } else if (discrepancy_mode == DiscrepancyMode::H_GAP) {
            const int best_rank_h = candidates[sorted_indices[0]].rank_h;
            for (size_t i = 0; i < sorted_indices.size(); ++i) {
                const int candidate_index = sorted_indices[i];
                const int candidate_h = candidates[candidate_index].rank_h;
                long long gap = static_cast<long long>(candidate_h) -
                                static_cast<long long>(best_rank_h);
                if (gap < 0)
                    gap = 0;
                if (gap > static_cast<long long>(numeric_limits<int>::max()))
                    gap = numeric_limits<int>::max();
                discrepancies[candidate_index] = static_cast<int>(gap);
            }
        } else {
            discrepancies[sorted_indices[0]] = 0;
        }

        for (size_t i = 0; i < candidates.size(); ++i) {
            const SuccessorCandidate &candidate = candidates[i];
            OperatorProxy op = task_proxy.get_operators()[candidate.op_id];
            SearchNode succ_node = candidate.succ_node;
            State succ_state = candidate.succ_state;

            int discrepancy = discrepancies[i];
            int succ_total_discrepancy = recorded_discrepancy + discrepancy;

            if (succ_node.is_new()) {
                succ_node.open_new_node(node, op, get_adjusted_cost(op));
            } else {
                if (succ_node.is_closed()) {
                    if (reopen_closed_nodes) {
                        statistics.inc_reopened();
                        succ_node.reopen_closed_node(
                            node, op, get_adjusted_cost(op));
                    } else {
                        succ_node.update_closed_node_parent(
                            node, op, get_adjusted_cost(op));
                        continue;
                    }
                } else {
                    succ_node.update_open_node_parent(
                        node, op, get_adjusted_cost(op));
                }
            }

            state_discrepancy[succ_state] = succ_total_discrepancy;
            push_open(succ_state, succ_total_discrepancy, succ_node.get_g());
        }

        return IN_PROGRESS;
    }

    return has_incumbent ? SOLVED : FAILED;
}

}
