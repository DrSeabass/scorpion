#include "rrd_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../evaluator.h"
#include "../plan_manager.h"
#include "../pruning_method.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <cassert>
#include <cmath>
#include <iterator>

using namespace std;

namespace rrd_search {

// The geometric distance correction d^ = d_base / (1 - eps_d) diverges as eps_d
// approaches 1; clamp the per-step distance error below 1 so it stays finite,
// and cap the resulting estimates so they cannot overflow an int frontier key.
static constexpr double MAX_DISTANCE_ERROR = 1.0 - 1e-3;
static constexpr double VALUE_CAP = 1e8;

static int clamp_to_int(double value) {
    if (value < 0.0)
        value = 0.0;
    if (value > VALUE_CAP)
        value = VALUE_CAP;
    return static_cast<int>(lround(value));
}

RRDSearch::RRDSearch(
    const shared_ptr<Evaluator> &h, const shared_ptr<Evaluator> &hhat,
    const shared_ptr<Evaluator> &dhat, double w, bool debias,
    const shared_ptr<PruningMethod> &pruning, OperatorCost cost_type, int bound,
    double max_time, const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      w(w),
      debias(debias),
      h_evaluator(h),
      hhat_evaluator(hhat),
      dhat_evaluator(dhat ? dhat : hhat),
      pruning_method(pruning),
      next_seq(0) {
    if (w < 1.0) {
        cerr << "RRDSearch: suboptimality bound w must be >= 1." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void RRDSearch::initialize() {
    log << "Conducting round-robin on distance (RR-D) with w = " << w
        << (debias ? ", path-based error correction on" : "")
        << ", (real) bound = " << bound << endl;

    assert(h_evaluator);
    assert(hhat_evaluator);
    assert(dhat_evaluator);

    set<Evaluator *> evals;
    h_evaluator->get_path_dependent_evaluators(evals);
    hhat_evaluator->get_path_dependent_evaluators(evals);
    dhat_evaluator->get_path_dependent_evaluators(evals);
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    focal_boundary = cleanup.end();

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    int h = eval_context.get_evaluator_value_or_infinity(h_evaluator.get());
    int hhat =
        eval_context.get_evaluator_value_or_infinity(hhat_evaluator.get());
    int dhat =
        eval_context.get_evaluator_value_or_infinity(dhat_evaluator.get());
    bool is_dead_end = (h == EvaluationResult::INFTY &&
                        h_evaluator->dead_ends_are_reliable()) ||
                       (hhat == EvaluationResult::INFTY &&
                        hhat_evaluator->dead_ends_are_reliable()) ||
                       (dhat == EvaluationResult::INFTY &&
                        dhat_evaluator->dead_ends_are_reliable());

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

        register_node(initial_state, h, hhat, dhat, -1, 0);
        frontier_insert(initial_state);
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void RRDSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

void RRDSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(h_evaluator.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

// Register a freshly generated `state`. `parent_seq` < 0 marks the root; the
// error accumulators are inherited from the parent plus this edge's single-step
// error (see apply_correction for how they debias the base estimates).
int RRDSearch::register_node(
    const State &state, int h, int hhat_base, int dhat_base, int parent_seq,
    int step_cost) {
    int seq = next_seq++;
    state_seq[state] = seq;
    seq_to_state.push_back(state.get_id());

    SeqInfo n;
    n.h = h;
    n.hhat_base = hhat_base;
    n.dhat_base = dhat_base;
    if (parent_seq < 0) {
        n.errh_sum = 0;
        n.errd_sum = 0;
        n.path_len = 0;
    } else {
        const SeqInfo &p = info[parent_seq];
        n.errh_sum = p.errh_sum + (step_cost + hhat_base - p.hhat_base);
        n.errd_sum = p.errd_sum + (1 + dhat_base - p.dhat_base);
        n.path_len = p.path_len + 1;
    }
    info.push_back(n);
    apply_correction(seq);
    return seq;
}

// A cheaper path to a known state reached the node via a new parent: recompute
// its path-based error accumulators (and thus its debiased estimates) from that
// parent. The base estimates are intrinsic to the state and stay put.
void RRDSearch::reparent_node(int seq, int parent_seq, int step_cost) {
    const SeqInfo &p = info[parent_seq];
    SeqInfo &n = info[seq];
    n.errh_sum = p.errh_sum + (step_cost + n.hhat_base - p.hhat_base);
    n.errd_sum = p.errd_sum + (1 + n.dhat_base - p.dhat_base);
    n.path_len = p.path_len + 1;
    apply_correction(seq);
}

// Set the effective estimates hhat/dhat used in the frontier keys. Without
// debiasing they are the base estimates; with it they are corrected by the mean
// single-step error along the node's path (Thayer, Dionne & Ruml 2011).
void RRDSearch::apply_correction(int seq) {
    SeqInfo &n = info[seq];
    if (!debias || n.path_len == 0) {
        n.hhat = n.hhat_base;
        n.dhat = n.dhat_base;
        return;
    }
    double inv_len = 1.0 / static_cast<double>(n.path_len);
    double eps_d = static_cast<double>(n.errd_sum) * inv_len;
    double eps_h = static_cast<double>(n.errh_sum) * inv_len;
    // The corrections only make sense as inflation; clamp away negative error
    // (which would deflate the estimate below the base) and the divergent tail.
    if (eps_d < 0.0)
        eps_d = 0.0;
    if (eps_d > MAX_DISTANCE_ERROR)
        eps_d = MAX_DISTANCE_ERROR;
    if (eps_h < 0.0)
        eps_h = 0.0;
    double dhat = static_cast<double>(n.dhat_base) / (1.0 - eps_d);
    double hhat = static_cast<double>(n.hhat_base) + dhat * eps_h;
    n.dhat = clamp_to_int(dhat);
    n.hhat = clamp_to_int(hhat);
}

// The frontier key of a node in each of the three lists, computed from its
// current g and (possibly debiased) estimates. Callers must insert and erase a
// node with the same g/estimates so the stored keys stay consistent.
RRDSearch::Entry RRDSearch::cleanup_entry(int seq) {
    int g = search_space.get_node(
                             state_registry.lookup_state(seq_to_state[seq]))
                .get_g();
    return {g + info[seq].h, seq};
}

RRDSearch::Entry RRDSearch::focal_d_entry(int seq) {
    return {info[seq].dhat, seq};
}

RRDSearch::Entry RRDSearch::focal_fhat_entry(int seq) {
    int g = search_space.get_node(
                             state_registry.lookup_state(seq_to_state[seq]))
                .get_g();
    return {g + info[seq].hhat, seq};
}

// A cleanup entry is admitted into the focal lists iff it sorts before the
// boundary in `cleanup`'s (f, seq) ordering, i.e. it is part of the current
// focal prefix { f <= w * f_min }.
bool RRDSearch::is_before_boundary(const Entry &e) const {
    return focal_boundary == cleanup.end() || e < *focal_boundary;
}

// Insert a node into all three lists, mirroring into the focal lists iff it
// lands in the current focal prefix. sync_focal() later fixes the boundary to
// the threshold.
void RRDSearch::frontier_insert(const State &state) {
    int seq = state_seq[state];
    Entry e_cleanup = cleanup_entry(seq);
    cleanup.insert(e_cleanup);
    if (is_before_boundary(e_cleanup)) {
        focal_d.insert(focal_d_entry(seq));
        focal_fhat.insert(focal_fhat_entry(seq));
    }
}

// Remove a node from all three lists, keeping the focal lists and the boundary
// consistent. If the node is the boundary element it is not in focal, but the
// boundary iterator must be advanced before the erase invalidates it. Must be
// called with the node's current g/estimates (i.e. before they change).
void RRDSearch::frontier_erase(const State &state) {
    int seq = state_seq[state];
    Entry e_cleanup = cleanup_entry(seq);
    if (focal_boundary != cleanup.end() && *focal_boundary == e_cleanup) {
        ++focal_boundary;
        cleanup.erase(e_cleanup);
        return;
    }
    bool in_focal = is_before_boundary(e_cleanup);
    cleanup.erase(e_cleanup);
    if (in_focal) {
        focal_d.erase(focal_d_entry(seq));
        focal_fhat.erase(focal_fhat_entry(seq));
    }
}

// Move the focal boundary so that both focal lists hold exactly
// { n in cleanup : f(n) <= w * f_min }.
void RRDSearch::sync_focal() {
    assert(!cleanup.empty());
    double threshold = w * static_cast<double>(cleanup.begin()->first);
    // Grow: pull in cleanup nodes newly within the threshold.
    while (focal_boundary != cleanup.end() &&
           static_cast<double>(focal_boundary->first) <= threshold) {
        int seq = focal_boundary->second;
        focal_d.insert(focal_d_entry(seq));
        focal_fhat.insert(focal_fhat_entry(seq));
        ++focal_boundary;
    }
    // Shrink: drop focal nodes that now exceed the threshold.
    while (focal_boundary != cleanup.begin()) {
        auto prev = std::prev(focal_boundary);
        if (static_cast<double>(prev->first) <= threshold)
            break;
        int seq = prev->second;
        focal_d.erase(focal_d_entry(seq));
        focal_fhat.erase(focal_fhat_entry(seq));
        focal_boundary = prev;
    }
}

// Round-robin selection: rotate through the distance, f^ and cleanup queues in
// lockstep with the expansion counter. Return the seq of the node to expand.
int RRDSearch::select_node() {
    sync_focal();
    // w >= 1 keeps f_min <= w * f_min, so the cleanup minimum is always in
    // focal; hence both focal lists are non-empty whenever cleanup is.
    assert(!cleanup.empty() && !focal_d.empty() && !focal_fhat.empty());

    int phase = statistics.get_expanded() % 3;
    if (phase == 0)
        return focal_d.begin()->second; // best_d^
    else if (phase == 1)
        return focal_fhat.begin()->second; // best_f^
    else
        return cleanup.begin()->second; // best_f = f_min
}

void RRDSearch::expand(const State &state) {
    SearchNode node = search_space.get_node(state);
    node.close();
    statistics.inc_expanded();

    int g = node.get_g();
    int parent_seq = state_seq[state];

    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(state, applicable_ops);
    pruning_method->prune_operators(state, applicable_ops);

    for (OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];

        int step_cost = get_adjusted_cost(op);
        int succ_g = g + step_cost;
        // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for
        // NORMAL cost type. Skip successors that cannot meet the cost bound.
        if (g + op.get_cost() >= bound)
            continue;

        State succ_state = state_registry.get_successor_state(state, op);
        statistics.inc_generated();

        for (Evaluator *evaluator : path_dependent_evaluators) {
            evaluator->notify_state_transition(state, op_id, succ_state);
        }

        SearchNode succ_node = search_space.get_node(succ_state);

        int h = 0;
        int hhat = 0;
        int dhat = 0;
        bool is_new = succ_node.is_new();
        bool is_goal = task_properties::is_goal_state(task_proxy, succ_state);
        if (is_new) {
            if (is_goal) {
                // Goals have zero cost- and distance-to-go; expose them on the
                // frontier so select_node can pick (and thereby return) them
                // when they fall within the bound.
                h = hhat = dhat = 0;
                statistics.inc_evaluated_states();
            } else {
                EvaluationContext succ_eval_context(
                    succ_state, succ_g, false, &statistics);
                statistics.inc_evaluated_states();
                h = succ_eval_context.get_evaluator_value_or_infinity(
                    h_evaluator.get());
                hhat = succ_eval_context.get_evaluator_value_or_infinity(
                    hhat_evaluator.get());
                dhat = succ_eval_context.get_evaluator_value_or_infinity(
                    dhat_evaluator.get());
                bool succ_dead_end =
                    (h == EvaluationResult::INFTY &&
                     h_evaluator->dead_ends_are_reliable()) ||
                    (hhat == EvaluationResult::INFTY &&
                     hhat_evaluator->dead_ends_are_reliable()) ||
                    (dhat == EvaluationResult::INFTY &&
                     dhat_evaluator->dead_ends_are_reliable());
                if (succ_dead_end) {
                    succ_node.mark_as_dead_end();
                    statistics.inc_dead_ends();
                    continue;
                }
                if (search_progress.check_progress(succ_eval_context)) {
                    statistics.print_checkpoint_line(succ_g);
                }
            }
        }

        if (is_new) {
            succ_node.open_new_node(node, op, step_cost);
            register_node(succ_state, h, hhat, dhat, parent_seq, step_cost);
            frontier_insert(succ_state);
        } else if (succ_node.is_dead_end()) {
            continue;
        } else if (succ_g < succ_node.get_g()) {
            // A strictly cheaper path to a known state. Re-key it on the
            // frontier (or reopen it if it was closed) and, when debiasing,
            // recompute its path-based error from the new parent. Duplicates
            // without a lower g are dropped.
            if (succ_node.is_open()) {
                frontier_erase(succ_state);
                succ_node.update_open_node_parent(node, op, step_cost);
                reparent_node(state_seq[succ_state], parent_seq, step_cost);
                frontier_insert(succ_state);
            } else {
                assert(succ_node.is_closed());
                statistics.inc_reopened();
                succ_node.reopen_closed_node(node, op, step_cost);
                reparent_node(state_seq[succ_state], parent_seq, step_cost);
                frontier_insert(succ_state);
            }
        }
    }
}

SearchStatus RRDSearch::step() {
    if (found_solution() && bound == 0)
        return SOLVED;

    if (cleanup.empty()) {
        log << "Open list is empty -- no solution!" << endl;
        return FAILED;
    }

    int seq = select_node();
    State state = state_registry.lookup_state(seq_to_state[seq]);

    // A selected goal is provably within the w-bound (a goal has h = 0, so
    // f = g <= w * f_min <= w * g(opt)): return it.
    if (task_properties::is_goal_state(task_proxy, state)) {
        Plan plan =
            search_space.trace_path(task_proxy, successor_generator, state);
        set_plan(plan);
        log << "RR-D: solution found with cost "
            << calculate_plan_cost(plan, task_proxy) << endl;
        return SOLVED;
    }

    frontier_erase(state);
    expand(state);
    return IN_PROGRESS;
}

}
