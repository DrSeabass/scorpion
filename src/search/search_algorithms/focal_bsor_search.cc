#include "focal_bsor_search.h"

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

using namespace std;

namespace focal_bsor_search {

FocalBSORSearch::FocalBSORSearch(
    const shared_ptr<Evaluator> &eval, const shared_ptr<Evaluator> &dist,
    double w, double aspect, bool round_robin, bool focal_expand_remainder,
    const shared_ptr<PruningMethod> &pruning, OperatorCost cost_type, int bound,
    double max_time, const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      w(w),
      aspect(aspect),
      round_robin(round_robin),
      focal_expand_remainder(focal_expand_remainder),
      eval(eval),
      dist(dist ? dist : eval),
      pruning_method(pruning),
      delta_down(1.0),
      delta_across(1.0),
      iteration(1),
      level(0),
      f_min_max(0),
      bound_ready(false),
      next_seq(0) {
    if (w < 1.0) {
        cerr << "FocalBSORSearch: suboptimality bound w must be >= 1." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (aspect <= 0.0) {
        cerr << "FocalBSORSearch: aspect must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void FocalBSORSearch::initialize() {
    log << "Conducting focal " << (round_robin ? "round-robin " : "")
        << "bounded-suboptimal rectangle search with w = " << w
        << ", aspect = " << aspect
        << ", focal_expand_remainder = " << focal_expand_remainder
        << ", (real) bound = " << bound << endl;

    assert(eval);
    assert(dist);

    // (delta_down, delta_across) = (a, 1) if a >= 1 else (1, 1/a).
    if (aspect >= 1.0) {
        delta_down = aspect;
        delta_across = 1.0;
    } else {
        delta_down = 1.0;
        delta_across = 1.0 / aspect;
    }
    iteration = 1;
    level = 0;

    set<Evaluator *> evals;
    eval->get_path_dependent_evaluators(evals);
    dist->get_path_dependent_evaluators(evals);
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    int h = eval_context.get_evaluator_value_or_infinity(eval.get());
    int d = eval_context.get_evaluator_value_or_infinity(dist.get());
    bool is_dead_end =
        (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) ||
        (d == EvaluationResult::INFTY && dist->dead_ends_are_reliable());

    if (is_dead_end) {
        log << "Initial state is a dead end." << endl;
    } else if (task_properties::is_goal_state(task_proxy, initial_state)) {
        log << "Initial state is a goal." << endl;
        Plan empty_plan;
        set_plan(empty_plan);
        bound = 0;
    } else {
        if (search_progress.check_progress(eval_context)) {
            statistics.print_checkpoint_line(0);
        }
        start_evaluator_statistics(eval_context);

        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();

        int seq = assign_seq(initial_state);
        node_de[initial_state] = 0;
        node_h[initial_state] = h;
        node_d[initial_state] = d;

        ensure_level(0);
        open.clear();
        // Seed the monotone bound with the initial f so the initial node is
        // bucketed into focal (f = h <= w * h for w >= 1).
        f_min_max = h;
        bound_ready = true;
        frontier_insert(initial_state);
        (void)seq;
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void FocalBSORSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

void FocalBSORSearch::start_evaluator_statistics(
    EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

int FocalBSORSearch::assign_seq(const State &state) {
    int seq = next_seq++;
    node_seq[state] = seq;
    seq_to_state.push_back(state.get_id());
    return seq;
}

void FocalBSORSearch::ensure_level(int idx) {
    while (static_cast<int>(focal.size()) <= idx) {
        focal.push_back(set<Entry>());
        remainder.push_back(set<Entry>());
        ec.push_back(0);
    }
}

bool FocalBSORSearch::has_non_empty_rect() const {
    // Count both sub-queues: while open is non-empty some node remains to be
    // promoted/expanded, and iterating must continue until it is reached.
    for (const set<Entry> &bucket : focal) {
        if (!bucket.empty())
            return true;
    }
    for (const set<Entry> &bucket : remainder) {
        if (!bucket.empty())
            return true;
    }
    return false;
}

// Running max of the open list's min f. Every open f_min lower-bounds C* for
// admissible eval, so their max is a tighter monotone lower bound.
void FocalBSORSearch::refresh_bound() {
    if (!open.empty()) {
        int f_min = open.begin()->first;
        f_min_max = bound_ready ? max(f_min_max, f_min) : f_min;
        bound_ready = true;
    }
}

double FocalBSORSearch::threshold() const {
    return w * static_cast<double>(f_min_max);
}

// Move now-in-bound nodes (f <= w * f_min_max) from a level's remainder queue
// into its focal queue. The remainder is f-ordered, so this stops at the first
// node still out of bound.
void FocalBSORSearch::promote_level(int idx) {
    if (idx >= static_cast<int>(remainder.size()))
        return;
    set<Entry> &rem = remainder[idx];
    double t = threshold();
    while (!rem.empty()) {
        auto it = rem.begin();
        int f = it->first;
        if (bound_ready && static_cast<double>(f) > t)
            break;
        int seq = it->second;
        rem.erase(it);
        State s = state_registry.lookup_state(seq_to_state[seq]);
        focal[idx].insert({node_d[s], seq});
        node_in_focal[s] = true;
    }
}

// Insert a frontier node into open (keyed on f) and into either the focal
// (keyed on d) or remainder (keyed on f) sub-queue of its depth level,
// according to whether it is within the current bound. The invariant is that a
// node lives in open and in exactly one sub-queue, or in neither.
void FocalBSORSearch::frontier_insert(const State &state) {
    int seq = node_seq[state];
    int de = node_de[state];
    int g = search_space.get_node(state).get_g();
    int f = g + node_h[state];
    open.insert({f, seq});
    ensure_level(de);
    bool in_focal = !bound_ready || static_cast<double>(f) <= threshold();
    if (in_focal) {
        focal[de].insert({node_d[state], seq});
    } else {
        remainder[de].insert({f, seq});
    }
    node_in_focal[state] = in_focal;
}

// Remove a frontier node from open and from whichever sub-queue holds it. Must
// be called with the node's current g/de (i.e. before those are changed) so the
// stored keys match.
void FocalBSORSearch::frontier_erase(const State &state) {
    int seq = node_seq[state];
    int de = node_de[state];
    int g = search_space.get_node(state).get_g();
    int f = g + node_h[state];
    open.erase({f, seq});
    if (node_in_focal[state]) {
        focal[de].erase({node_d[state], seq});
    } else {
        remainder[de].erase({f, seq});
    }
}

// NEXT (Algorithm 4, lines 24-29): advance `iteration`/`level` to the next
// rectangle cell that has an expandable node, or report that none remain. A
// level is expandable if, after promotion, its focal queue is non-empty (or,
// with focal_expand_remainder, its remainder queue is non-empty).
bool FocalBSORSearch::advance_rectangle() {
    while (true) {
        if (level < static_cast<int>(focal.size())) {
            promote_level(level);
            bool expandable = !focal[level].empty() ||
                              (focal_expand_remainder &&
                               !remainder[level].empty());
            if (expandable && ec[level] < iteration * delta_across) {
                return true;
            }
        }
        if (level < iteration * delta_down - 1) {
            ++level;
        } else if (has_non_empty_rect()) {
            ++iteration;
            level = 0;
        } else {
            return false;
        }
    }
}

// EXPAND (Algorithm 5). `state` has already been removed from the frontier.
void FocalBSORSearch::expand(const State &state) {
    SearchNode node = search_space.get_node(state);
    node.close();
    statistics.inc_expanded();

    int g = node.get_g();
    int de = node_de[state];

    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(state, applicable_ops);
    pruning_method->prune_operators(state, applicable_ops);

    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];

        int succ_g = g + get_adjusted_cost(op);
        // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for
        // NORMAL cost type.
        if (g + op.get_cost() >= bound)
            continue;

        State succ_state = state_registry.get_successor_state(state, op);
        statistics.inc_generated();

        for (Evaluator *evaluator : path_dependent_evaluators) {
            evaluator->notify_state_transition(state, op_id, succ_state);
        }

        // Handle goals directly (Algorithm 5, line 5): record the incumbent if
        // this path improves it and do not place the goal on the frontier. We
        // check goals by cost first so an inadmissible eval cannot prune an
        // improving goal via the f-bound below.
        if (task_properties::is_goal_state(task_proxy, succ_state)) {
            try_improve_incumbent(state, op_id, succ_g);
            continue;
        }

        SearchNode succ_node = search_space.get_node(succ_state);

        int h;
        int d;
        bool is_new = succ_node.is_new();
        if (is_new) {
            EvaluationContext succ_eval_context(
                succ_state, succ_g, false, &statistics);
            statistics.inc_evaluated_states();
            h = succ_eval_context.get_evaluator_value_or_infinity(eval.get());
            d = succ_eval_context.get_evaluator_value_or_infinity(dist.get());
            bool succ_dead_end = (h == EvaluationResult::INFTY &&
                                  eval->dead_ends_are_reliable()) ||
                                 (d == EvaluationResult::INFTY &&
                                  dist->dead_ends_are_reliable());
            if (succ_dead_end) {
                succ_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                continue;
            }
            if (search_progress.check_progress(succ_eval_context)) {
                statistics.print_checkpoint_line(succ_g);
            }
        } else {
            if (succ_node.is_dead_end())
                continue;
            h = node_h[succ_state];
            d = node_d[succ_state];
        }

        int f = succ_g + h;
        // Prune nodes that cannot improve on the incumbent (Algorithm 5, line
        // 4).
        if (found_solution() && f >= bound)
            continue;

        if (is_new) {
            succ_node.open_new_node(node, op, get_adjusted_cost(op));
            assign_seq(succ_state);
            node_de[succ_state] = de + 1;
            node_h[succ_state] = h;
            node_d[succ_state] = d;
            frontier_insert(succ_state);
        } else if (succ_g < succ_node.get_g()) {
            // A strictly cheaper path to a known state (Algorithm 5, lines
            // 10-21). Duplicates without a lower g are pruned (the else
            // branch).
            if (succ_node.is_open()) {
                frontier_erase(succ_state);
                succ_node.update_open_node_parent(
                    node, op, get_adjusted_cost(op));
                node_de[succ_state] = de + 1;
                frontier_insert(succ_state);
            } else {
                assert(succ_node.is_closed());
                statistics.inc_reopened();
                succ_node.reopen_closed_node(node, op, get_adjusted_cost(op));
                node_de[succ_state] = de + 1;
                frontier_insert(succ_state);
            }
        }
    }
}

// Record the plan reaching `parent_state` then applying `op_id` as the new
// incumbent iff it is cheaper than the current one (Algorithm 5, line 5).
bool FocalBSORSearch::try_improve_incumbent(
    const State &parent_state, OperatorID op_id, int succ_g) {
    if (found_solution() && succ_g >= bound)
        return false;

    Plan plan =
        search_space.trace_path(task_proxy, successor_generator, parent_state);
    plan.push_back(op_id);
    int cost = calculate_plan_cost(plan, task_proxy);

    if (found_solution() && cost >= bound)
        return false;

    set_plan(plan);
    bound = cost;
    log << "Focal BSOR: improved incumbent with cost " << cost << endl;
    return true;
}

SearchStatus FocalBSORSearch::step() {
    if (found_solution() && bound == 0)
        return SOLVED;

    if (open.empty()) {
        if (found_solution()) {
            log << "Open list is empty -- bounded-suboptimal solution found."
                << endl;
            return SOLVED;
        }
        log << "Open list is empty -- no solution!" << endl;
        return FAILED;
    }

    refresh_bound();

    // Termination (Algorithm 4, line 6): stop once the incumbent is provably
    // within the suboptimality bound, i.e. w * f_min_max >= g(incumbent).
    if (found_solution()) {
        if (threshold() >= static_cast<double>(bound)) {
            log << "Incumbent proven within suboptimality bound." << endl;
            return SOLVED;
        }
    }

    // Round-Robin variant: one lowest-f expansion before the rectangle
    // expansion (Algorithm 4, lines 7-13).
    if (round_robin) {
        while (true) {
            if (open.empty())
                return found_solution() ? SOLVED : FAILED;
            Entry entry = *open.begin();
            State n = state_registry.lookup_state(seq_to_state[entry.second]);
            SearchNode n_node = search_space.get_node(n);
            // Remove from open and from whichever sub-queue holds it.
            frontier_erase(n);

            int f_n = n_node.get_g() + node_h[n];
            if (f_n < bound) {
                expand(n);
                break;
            }
            // Cannot improve on the incumbent; discard it (close without
            // expanding).
            n_node.close();
        }
        // An open expansion may have raised f_min; refresh so the rectangle
        // promotion below sees the tighter bound.
        refresh_bound();
    }

    // Rectangle expansion (Algorithm 4, lines 14-22). advance_rectangle has
    // promoted the current level, so focal[level] (or, with the fallback,
    // remainder[level]) is non-empty.
    State n = state_registry.get_initial_state();
    bool have_node = false;
    while (!have_node) {
        if (!advance_rectangle())
            return found_solution() ? SOLVED : FAILED;
        int seq;
        if (!focal[level].empty()) {
            seq = focal[level].begin()->second;
        } else {
            assert(focal_expand_remainder && !remainder[level].empty());
            seq = remainder[level].begin()->second;
        }
        n = state_registry.lookup_state(seq_to_state[seq]);
        frontier_erase(n);

        SearchNode n_node = search_space.get_node(n);
        int f_n = n_node.get_g() + node_h[n];
        if (f_n < bound) {
            have_node = true;
        } else {
            n_node.close();
        }
    }

    // Lazily create the next depth level just before it may be needed
    // (Algorithm 4, lines 19-20), then count this expansion (line 21).
    if (static_cast<double>(level) >= (iteration - 1) * delta_down)
        ensure_level(level + 1);
    ++ec[level];
    expand(n);

    return IN_PROGRESS;
}

}
