#include "adaptive_rectangle_search.h"

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

namespace adaptive_rectangle_search {

// Safety rails on the live aspect ratio (not tuning knobs): keep a in a range
// where iteration * delta stays well within int and the schedule never
// degenerates to a single shape forever.
static const double ASPECT_FLOOR = 1.0 / 1024.0;
static const double ASPECT_CEILING = 1024.0;

AdaptiveRectangleSearch::AdaptiveRectangleSearch(
    const shared_ptr<Evaluator> &eval, bool reopen_closed, bool anytime,
    bool log_aspect, const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      log_aspect(log_aspect),
      eval(eval),
      pruning_method(pruning),
      aspect(1.0),
      delta_down(1.0),
      delta_across(1.0),
      chain_intact_votes(0),
      chain_broken_votes(0),
      spine_level(-1),
      iteration(1),
      level(0),
      next_seq(0) {
}

void AdaptiveRectangleSearch::recompute_deltas() {
    // (delta_down, delta_across) = (a, 1) if a >= 1 else (1, 1/a).
    if (aspect >= 1.0) {
        delta_down = aspect;
        delta_across = 1.0;
    } else {
        delta_down = 1.0;
        delta_across = 1.0 / aspect;
    }
}

void AdaptiveRectangleSearch::initialize() {
    log << "Conducting adaptive rectangle search (dynamic aspect ratio), "
        << "(real) bound = " << bound << endl;

    assert(eval);

    if (log_aspect) {
        aspect_log_file.open("adaptive_rectangle_aspect.csv");
        aspect_log_file << "expansions,aspect,mindepth,maxdepth\n";
        {
            int mind, maxd;
            current_depth_range(mind, maxd);
            aspect_log_file << statistics.get_expanded() << "," << 1.0 << ","
                            << mind << "," << maxd << "\n";
        }
        ++log_rows_written;
        aspect_solution_file.open("adaptive_rectangle_solutions.csv");
        aspect_solution_file << "log_element\n";
    }

    aspect = 1.0;
    recompute_deltas();
    iteration = 1;
    level = 0;
    chain_intact_votes = 0;
    chain_broken_votes = 0;
    spine_level = -1;

    set<Evaluator *> evals;
    eval->get_path_dependent_evaluators(evals);
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    int h = eval_context.get_evaluator_value_or_infinity(eval.get());
    bool is_dead_end =
        h == EvaluationResult::INFTY && eval->dead_ends_are_reliable();

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

        assign_seq(initial_state);
        node_de[initial_state] = 0;
        node_h[initial_state] = h;

        ensure_level(0);
        frontier_insert(initial_state);
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void AdaptiveRectangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

void AdaptiveRectangleSearch::start_evaluator_statistics(
    EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

int AdaptiveRectangleSearch::assign_seq(const State &state) {
    int seq = next_seq++;
    node_seq[state] = seq;
    seq_to_state.push_back(state.get_id());
    return seq;
}

void AdaptiveRectangleSearch::ensure_level(int idx) {
    while (static_cast<int>(rect.size()) <= idx) {
        rect.push_back(set<Entry>());
        ec.push_back(0);
    }
}

bool AdaptiveRectangleSearch::has_non_empty_rect() const {
    for (const set<Entry> &bucket : rect) {
        if (!bucket.empty())
            return true;
    }
    return false;
}

void AdaptiveRectangleSearch::current_depth_range(int &mind, int &maxd) const {
    int n = static_cast<int>(rect.size());
    mind = maxd = -1;
    for (int i = 0; i < n; ++i) {
        if (!rect[i].empty()) {
            mind = i;
            break;
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        if (!rect[i].empty()) {
            maxd = i;
            break;
        }
    }
}

// Insert a frontier node into its rectangle depth bucket, keyed on the eval
// value used for the within-depth ordering.
void AdaptiveRectangleSearch::frontier_insert(const State &state) {
    int seq = node_seq[state];
    int de = node_de[state];
    ensure_level(de);
    rect[de].insert({node_h[state], seq});
}

// Remove a frontier node from its rectangle depth bucket. Must be called with
// the node's current de/h (i.e. before those are changed) so the stored key
// matches.
void AdaptiveRectangleSearch::frontier_erase(const State &state) {
    int seq = node_seq[state];
    int de = node_de[state];
    rect[de].erase({node_h[state], seq});
}

// Best-first-chain ratchet, applied at each completed rectangle sweep. Rotate
// deeper (a *= 2) when chain-intact votes strictly dominate, wider (a /= 2)
// when chain-broken votes strictly dominate, hold on a tie or no data. Then
// reset the tallies and the per-sweep spine tracker and recompute the split.
void AdaptiveRectangleSearch::apply_aspect_ratchet() {
    if (chain_intact_votes > chain_broken_votes) {
        aspect = min(aspect * 2.0, ASPECT_CEILING);
    } else if (chain_broken_votes > chain_intact_votes) {
        aspect = max(aspect / 2.0, ASPECT_FLOOR);
    }
    chain_intact_votes = 0;
    chain_broken_votes = 0;
    spine_level = -1;
    recompute_deltas();

    if (log_aspect) {
        int mind, maxd;
        current_depth_range(mind, maxd);
        aspect_log_file << statistics.get_expanded() << "," << aspect << ","
                        << mind << "," << maxd << "\n";
        ++log_rows_written;
    }
}

// NEXT: advance `iteration`/`level` to the next rectangle cell that is
// eligible for expansion, or report that none remain. The aspect ratchet fires
// at each `iteration` boundary (a completed sweep).
bool AdaptiveRectangleSearch::advance_rectangle() {
    while (true) {
        if (level < static_cast<int>(rect.size()) && !rect[level].empty() &&
            ec[level] < iteration * delta_across) {
            return true;
        }
        if (level < iteration * delta_down - 1) {
            ++level;
        } else if (has_non_empty_rect()) {
            apply_aspect_ratchet();
            ++iteration;
            level = 0;
        } else {
            return false;
        }
    }
}

// EXPAND. `state` has already been removed from its rectangle bucket. Returns
// true iff the search should stop now (first solution found with anytime off).
// When `first_in_beam` (this is the best node expanded at its depth level this
// sweep), casts the best-first-chain vote for the aspect ratchet.
bool AdaptiveRectangleSearch::expand(const State &state, bool first_in_beam) {
    SearchNode node = search_space.get_node(state);
    node.close();
    statistics.inc_expanded();

    int g = node.get_g();
    int de = node_de[state];

    // Successors this node places on the next depth bucket (de + 1); used for
    // the best-first-chain vote below.
    vector<int> inserted_children;

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

        // Handle goals directly: record the incumbent if this path improves it
        // and do not place the goal on the frontier. We check goals by cost
        // first so an inadmissible eval cannot prune an improving goal via the
        // f-bound below.
        if (task_properties::is_goal_state(task_proxy, succ_state)) {
            bool improved = try_improve_incumbent(state, op_id, succ_g);
            if (improved && !anytime_search)
                return true;
            continue;
        }

        SearchNode succ_node = search_space.get_node(succ_state);

        int h;
        bool is_new = succ_node.is_new();
        if (is_new) {
            EvaluationContext succ_eval_context(
                succ_state, succ_g, false, &statistics);
            statistics.inc_evaluated_states();
            h = succ_eval_context.get_evaluator_value_or_infinity(eval.get());
            if (h == EvaluationResult::INFTY &&
                eval->dead_ends_are_reliable()) {
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
        }

        int f = succ_g + h;
        // Prune nodes that cannot improve on the incumbent.
        if (found_solution() && f >= bound)
            continue;

        if (is_new) {
            succ_node.open_new_node(node, op, get_adjusted_cost(op));
            assign_seq(succ_state);
            node_de[succ_state] = de + 1;
            node_h[succ_state] = h;
            frontier_insert(succ_state);
            inserted_children.push_back(node_seq[succ_state]);
        } else if (succ_g < succ_node.get_g()) {
            // A strictly cheaper path to a known state. Duplicates without a
            // lower g are pruned (no else branch).
            if (succ_node.is_open()) {
                frontier_erase(succ_state);
                succ_node.update_open_node_parent(
                    node, op, get_adjusted_cost(op));
                node_de[succ_state] = de + 1;
                frontier_insert(succ_state);
                inserted_children.push_back(node_seq[succ_state]);
            } else {
                assert(succ_node.is_closed());
                if (!reopen_closed_nodes)
                    continue;
                statistics.inc_reopened();
                succ_node.reopen_closed_node(node, op, get_adjusted_cost(op));
                node_de[succ_state] = de + 1;
                frontier_insert(succ_state);
                inserted_children.push_back(node_seq[succ_state]);
            }
        }
    }

    // Best-first-chain vote (only for the head of this level's beam): did the
    // node just expanded produce the node now at the front of the next depth's
    // bucket? If so the greedy best-first chain is intact (deepen); otherwise
    // the best path is scattering across the frontier (widen).
    if (first_in_beam) {
        int next = de + 1;
        bool chain_intact = false;
        if (next < static_cast<int>(rect.size()) && !rect[next].empty()) {
            int front_seq = rect[next].begin()->second;
            for (int child_seq : inserted_children) {
                if (child_seq == front_seq) {
                    chain_intact = true;
                    break;
                }
            }
        }
        if (chain_intact)
            ++chain_intact_votes;
        else
            ++chain_broken_votes;
    }

    return false;
}

// Record the plan reaching `parent_state` then applying `op_id` as the new
// incumbent iff it is cheaper than the current one.
bool AdaptiveRectangleSearch::try_improve_incumbent(
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
    // Mark the log element for the (upcoming) end of the sweep in which this
    // solution was found.
    if (log_aspect)
        aspect_solution_file << log_rows_written << "\n";
    log << "AdaptiveRectangleSearch: improved incumbent with cost " << cost
        << " (aspect " << aspect << ")" << endl;
    if (anytime_search) {
        plan_manager.save_plan(plan, task_proxy, true);
    }
    return true;
}

SearchStatus AdaptiveRectangleSearch::step() {
    if (found_solution() && bound == 0)
        return SOLVED;
    if (!anytime_search && found_solution())
        return SOLVED;

    if (!has_non_empty_rect()) {
        if (found_solution()) {
            log << "Rectangle is empty -- best solution found." << endl;
            return SOLVED;
        }
        log << "Rectangle is empty -- no solution!" << endl;
        return FAILED;
    }

    // Rectangle expansion: pick the next eligible node, skipping any that can
    // no longer improve the incumbent.
    State n = state_registry.get_initial_state();
    bool have_node = false;
    while (!have_node) {
        if (!advance_rectangle())
            return found_solution() ? SOLVED : FAILED;
        Entry entry = *rect[level].begin();
        rect[level].erase(rect[level].begin());
        n = state_registry.lookup_state(seq_to_state[entry.second]);
        SearchNode n_node = search_space.get_node(n);

        int f_n = n_node.get_g() + node_h[n];
        if (!found_solution() || f_n < bound) {
            have_node = true;
        } else {
            n_node.close();
        }
    }

    // This node is the head of its level's beam iff it is the first node
    // expanded at this depth in the current sweep (levels are served in
    // increasing order, and skipped/pruned nodes do not vote). Only the head
    // casts the best-first-chain vote (see expand()).
    bool first_in_beam = level > spine_level;
    if (first_in_beam)
        spine_level = level;

    // Lazily create the next depth level just before it may be needed, then
    // count this expansion.
    if (static_cast<double>(level) >= (iteration - 1) * delta_down)
        ensure_level(level + 1);
    ++ec[level];
    if (expand(n, first_in_beam))
        return SOLVED;

    return IN_PROGRESS;
}

}
