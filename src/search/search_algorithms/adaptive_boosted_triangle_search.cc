#include "adaptive_boosted_triangle_search.h"

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

namespace adaptive_boosted_triangle_search {

AdaptiveBoostedTriangleSearch::AdaptiveBoostedTriangleSearch(
    const vector<shared_ptr<Evaluator>> &evals,
    bool reopen_closed,
    bool anytime,
    Schedule schedule,
    int credit_boost,
    const vector<shared_ptr<Evaluator>> &preferred_evals,
    bool guide_by_pruning,
    const shared_ptr<Evaluator> &pruning_heuristic,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      schedule(schedule),
      credit_boost(credit_boost),
      guide_by_pruning(guide_by_pruning),
      evals(evals),
      num_lists(static_cast<int>(evals.size())),
      preferred_evals(preferred_evals),
      num_preferred(static_cast<int>(preferred_evals.size())),
      use_pruner_queue(guide_by_pruning && pruning_heuristic != nullptr),
      total_lists(
          static_cast<int>(evals.size()) + static_cast<int>(preferred_evals.size()) +
          (guide_by_pruning && pruning_heuristic != nullptr ? 1 : 0)),
      pruning_heuristic(pruning_heuristic),
      pruning_method(pruning) {
    if (evals.empty()) {
        cerr << "AdaptiveBoostedTriangleSearch: at least one guidance evaluator is required." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    // ICAPS-27 step 6: every preferred_evals entry must also be one of the
    // guidance evals (matched by identity), so its helpful list can share
    // its h-value/ranking without a second evaluation.
    preferred_source_index.reserve(preferred_evals.size());
    for (const shared_ptr<Evaluator> &pref_eval : preferred_evals) {
        int source = -1;
        for (int k = 0; k < num_lists; ++k) {
            if (evals[k].get() == pref_eval.get()) {
                source = k;
                break;
            }
        }
        if (source == -1) {
            cerr << "AdaptiveBoostedTriangleSearch: every preferred_evals entry "
                    "must also appear in evals." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        preferred_source_index.push_back(source);
    }
}

void AdaptiveBoostedTriangleSearch::initialize() {
    log << "Conducting adaptive-boosted triangle search with "
        << num_lists << " guidance heuristic(s)"
        << ", schedule = " << (schedule == Schedule::SWEEP ? "sweep" : "pop")
        << ", credit_boost = " << credit_boost
        << ", " << num_preferred << " preferred-operator (helpful) list(s)"
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

    EvaluationContext eval_context(initial_state, 0, true, &statistics, num_preferred > 0);
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
    // ICAPS-27 step 6: helpful lists share their h-value with their source
    // guidance list -- no extra evaluation, just a copy at the right index.
    for (int j = 0; j < num_preferred; ++j)
        initial_h.push_back(initial_h[preferred_source_index[j]]);
    if (use_pruner_queue) {
        // Seed the pruner queue from the admissible h of the initial state.
        int prune_h =
            eval_context.get_evaluator_value_or_infinity(pruning_heuristic.get());
        if (prune_h == EvaluationResult::INFTY &&
            pruning_heuristic->dead_ends_are_reliable())
            is_dead_end = true;
        initial_h.push_back(prune_h);
    }

    extend_layers(1);

    if (is_dead_end) {
        log << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context)) {
            statistics.print_checkpoint_line(0);
        }
        start_evaluator_statistics(eval_context);

        // ICAPS-27 step 6: cache the initial state's own preferred-operator
        // sets so they're ready when it is later popped for expansion.
        if (num_preferred > 0)
            cache_preferred_operators(initial_state, eval_context);

        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();
        if (task_properties::is_goal_state(task_proxy, initial_state)) {
            update_incumbent(initial_state);
        } else {
            // The initial state was not reached via any operator, so it is
            // never a member of a helpful list (nothing to be "preferred"
            // relative to) -- only its guidance/pruner entries are inserted.
            insert_successor(
                0, initial_state.get_id(), 0, initial_h,
                vector<bool>(num_preferred, false));
        }
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void AdaptiveBoostedTriangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();

    // ICAPS-27 progress credit (axis 1a, see icaps-27-plan.md): final
    // per-heuristic budgets for any layers still live when the search ended.
    if (log.is_at_least_debug()) {
        for (size_t i = 0; i < layers.size(); ++i) {
            log << "Adaptive-boosted triangle layer " << (depth_offset + static_cast<int>(i))
                << " final budgets:";
            for (int k = 0; k < total_lists; ++k)
                log << " " << layers[i].progress[k].budget;
            log << endl;
        }
    }
}

void AdaptiveBoostedTriangleSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    // Report f-value progress against the primary guidance heuristic.
    int value = eval_context.get_evaluator_value_or_infinity(evals[0].get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

void AdaptiveBoostedTriangleSearch::extend_layers(int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.emplace_back(total_lists);
    }
}

bool AdaptiveBoostedTriangleSearch::layer_empty(int layer) const {
    for (const OpenList &list : layers[layer].lists) {
        if (!list.empty())
            return false;
    }
    return true;
}

void AdaptiveBoostedTriangleSearch::recompute_max_active_layer() {
    while (max_active_layer >= 0 && layer_empty(max_active_layer)) {
        --max_active_layer;
    }
}

void AdaptiveBoostedTriangleSearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        log << "AdaptiveBoostedTriangleSearch: improved incumbent with cost " << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

bool AdaptiveBoostedTriangleSearch::evaluate_and_prepare_node(
    const State &state, SearchNode &node, int g,
    vector<int> &h_out, bool is_new_evaluation) {
    // ICAPS-27 step 6: request preferred operators on this context whenever
    // helpful lists are in use, so any evaluator queried below computes
    // them as a side effect of its normal h-value call -- no evaluator is
    // ever run a second time just to learn its preferred operators.
    EvaluationContext eval_context(state, g, false, &statistics, num_preferred > 0);
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

    if (num_preferred > 0)
        cache_preferred_operators(state, eval_context);

    if (is_new_evaluation && search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(node.get_g());
    }

    return true;
}

void AdaptiveBoostedTriangleSearch::cache_preferred_operators(
    const State &state, EvaluationContext &eval_context) {
    vector<vector<OperatorID>> prefs(num_preferred);
    for (int j = 0; j < num_preferred; ++j)
        prefs[j] = eval_context.get_preferred_operators(preferred_evals[j].get());
    preferred_op_cache[state] = std::move(prefs);
}

int AdaptiveBoostedTriangleSearch::select_credit_served(
    const Layer &layer, int round_robin_served) const {
    // ICAPS-27 step 6: helpful lists can be selectively empty (unlike
    // guidance/pruner lists, which always hold identical live content), so
    // only lists with live entries at this layer are eligible. If every
    // list happens to be empty, round_robin_served is returned unchanged --
    // the caller's list.empty() check will then correctly detect that the
    // whole layer is exhausted (guidance lists, a superset of everything
    // else, would be empty too in that case).
    int best = -1;
    for (int k = 0; k < total_lists; ++k) {
        if (layer.lists[k].empty())
            continue;
        if (best == -1 || layer.progress[k].budget > layer.progress[best].budget)
            best = k;
    }
    if (best == -1)
        return round_robin_served;
    // Prefer the round-robin's own candidate among ties, but only if it is
    // itself non-empty.
    if (!layer.lists[round_robin_served].empty() &&
        layer.progress[round_robin_served].budget == layer.progress[best].budget)
        return round_robin_served;
    return best;
}

int AdaptiveBoostedTriangleSearch::select_round_robin_served(
    const Layer &layer, int round_robin_served) const {
    // ICAPS-27 step 6: walk the round-robin order (wrapping) until a
    // non-empty list is found. At preferred_evals=[] this always returns
    // round_robin_served itself on the very first check (every guidance/
    // pruner list shares identical live content, so round_robin_served's
    // own list can only be empty when every list is) -- an exact reduction
    // to the pre-step-6 behavior.
    for (int offset = 0; offset < total_lists; ++offset) {
        int k = (round_robin_served + offset) % total_lists;
        if (!layer.lists[k].empty())
            return k;
    }
    return round_robin_served;
}

void AdaptiveBoostedTriangleSearch::record_expansion_credit(
    HeuristicProgress &hp, int h) const {
    if (hp.have_last_h && h < hp.last_h)
        hp.budget += credit_boost;
    hp.have_last_h = true;
    hp.last_h = h;
    --hp.budget;
}

void AdaptiveBoostedTriangleSearch::insert_successor(
    int layer, StateID id, int g, const vector<int> &hs,
    const vector<bool> &helpful_membership) {
    assert(layer >= 0);
    assert(static_cast<int>(hs.size()) == total_lists);
    assert(static_cast<int>(helpful_membership.size()) == num_preferred);
    if (layer >= static_cast<int>(layers.size())) {
        extend_layers(layer + 1 - static_cast<int>(layers.size()));
    }
    // Guidance lists always receive every successor.
    for (int k = 0; k < num_lists; ++k) {
        layers[layer].lists[k].push({id, hs[k], g});
    }
    // ICAPS-27 step 6: helpful lists only receive successors reached via
    // that heuristic's own preferred operator on the parent.
    for (int j = 0; j < num_preferred; ++j) {
        if (helpful_membership[j])
            layers[layer].lists[num_lists + j].push({id, hs[num_lists + j], g});
    }
    // The optional pruner list, like the guidance lists, receives every
    // successor.
    if (use_pruner_queue) {
        int pruner_index = total_lists - 1;
        layers[layer].lists[pruner_index].push({id, hs[pruner_index], g});
    }
    if (layer > max_active_layer)
        max_active_layer = layer;
}

SearchStatus AdaptiveBoostedTriangleSearch::step() {
    while (!layers.empty() && layer_empty(0)) {
        if (log.is_at_least_debug()) {
            log << "AdaptiveBoostedTriangleSearch: layer " << depth_offset
                << " drained, final budgets:";
            for (int k = 0; k < total_lists; ++k)
                log << " " << layers.front().progress[k].budget;
            log << endl;
        }
        layers.pop_front();
        ++depth_offset;
        --max_active_layer;
    }
    max_active_layer = max(max_active_layer, 0);

    if (layers.empty()) {
        if (found_solution()) {
            log << "All open lists are empty -- best solution found." << endl;
            return SOLVED;
        }
        log << "All open lists are empty -- no solution!" << endl;
        return FAILED;
    }

    // Adaptive per-step depth budget (ported as-is from
    // adaptive_triangle_search, always on -- see icaps-27-plan.md step 5).
    // Unspent budget carries between steps; each step starts with at least
    // one unit so forward progress is always possible even after a
    // depleted step.
    depth_budget = max(1, depth_budget);

    // SWEEP: one guidance list owns the entire cascade dive this step; the
    // served index rotates between steps. POP: the served index advances per
    // expansion (below), so successive expansions down the dive alternate
    // heuristics. Every live state is inserted into all N guidance (and
    // pruner) lists, so those lists being (live-)empty at a layer means the
    // layer is live-empty -- see select_round_robin_served /
    // select_credit_served for how step 6's sparser helpful lists changed
    // that reasoning.
    const int sweep_served = sweep_count % total_lists;

    // Adaptive depth-budget trend signal: a single step-scoped counter over
    // every expansion in the step, regardless of which guidance list served
    // it -- orthogonal to the per-list progress-credit budgets above.
    // last_expanded_h is preserved across skipped empty layers.
    int last_expanded_h = 0;
    bool have_last_h = false;
    // Number of new frontier layers this step has paid for so far; the
    // first extension of the step is always free (see the budget gate
    // below), only later ones are gated.
    int layers_added = 0;

    for (int i = 0; ; ++i) {
        // End the cascade as soon as we run off the end of the deque.
        if (i >= static_cast<int>(layers.size()))
            break;
        const int round_robin_served =
            (schedule == Schedule::SWEEP) ? sweep_served : pop_count % total_lists;
        // ICAPS-27 axis 1a x 2c (see icaps-27-plan.md): reselect the served
        // list at this layer boundary from its own budgets. credit_boost ==
        // 0 falls straight through to the round-robin selector -- no credit
        // consulted, no budgets touched below. Both selectors pick among
        // the layer's currently non-empty lists (step 6).
        const int served =
            (credit_boost == 0)
                ? select_round_robin_served(layers[i], round_robin_served)
                : select_credit_served(layers[i], round_robin_served);
        OpenList &list = layers[i].lists[served];
        if (list.empty())
            continue;

        // Drain ineligible entries (stale, dead-end, already closed) from
        // the top of the served list at layer i without committing to
        // expansion yet -- we don't want to pay the frontier-extension
        // budget until we know we have someone to expand here. The
        // expandable entry is identified but left on top until the budget
        // check passes.
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
            found_expandable = true;
            break;
        }
        if (!found_expandable)
            continue;

        // We have an expandable top. Ensure layer i+1 exists to receive its
        // successors; free if already in the deque, otherwise pay one
        // depth_budget unit. Every pass lays at least one new frontier
        // layer: the first extension of the step is granted unconditionally
        // (even with a depleted/negative budget), and the budget gate is
        // consulted only for extensions beyond that first one. If a later
        // extension can't be afforded, halt the cascade with the expandable
        // entry still in place for the next step.
        if (i + 1 >= static_cast<int>(layers.size())) {
            if (layers_added > 0 && depth_budget <= 0)
                break;
            --depth_budget;
            extend_layers(1);
            ++layers_added;
        }

        // Commit: actually pop the expandable entry.
        list.pop();
        if (layer_empty(i) && i == max_active_layer)
            recompute_max_active_layer();

        State state = state_registry.lookup_state(current.id);
        SearchNode node = search_space.get_node(state);

        node.close();
        statistics.inc_expanded();

        // ICAPS-27 progress credit (axis 1a, see icaps-27-plan.md). The
        // pruner list (when present) earns/spends tokens the same as any
        // guidance or helpful list. Skipped at credit_boost == 0, matching
        // the selection short-circuit above -- there is nothing to earn,
        // and leaving budgets at 0 keeps debug logging honest about the
        // mechanism being fully inert.
        if (credit_boost != 0)
            record_expansion_credit(layers[i].progress[served], current.h);

        // Adaptive depth-budget trend signal (see above): an informed
        // transition (h decreases relative to the previous expansion in
        // this step) refunds one unit, an uninformed one debits one (the
        // original symmetric +1/-1 rule, ported as-is).
        if (have_last_h) {
            if (current.h < last_expanded_h)
                ++depth_budget;
            else
                --depth_budget;
        }
        last_expanded_h = current.h;
        have_last_h = true;

        // ICAPS-27 step 6: consume this state's cached preferred-operator
        // sets (computed once, back when it was generated -- see
        // evaluate_and_prepare_node) so no evaluator is run a second time
        // just to learn which of its operators are preferred.
        vector<vector<OperatorID>> parent_preferred;
        if (num_preferred > 0)
            parent_preferred = std::move(preferred_op_cache[state]);

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

            // ICAPS-27 step 6: helpful lists share their h-value with their
            // source guidance list -- no extra evaluation, just a copy at
            // the right index. The guidance heuristics filled succ_h (size
            // num_lists); append the helpful copies, then the admissible
            // value (if the pruner queue is in use) so every list in
            // total_lists order is ranked.
            for (int j = 0; j < num_preferred; ++j)
                succ_h.push_back(succ_h[preferred_source_index[j]]);
            if (use_pruner_queue)
                succ_h.push_back(prune_h);

            vector<bool> helpful_membership(num_preferred);
            for (int j = 0; j < num_preferred; ++j) {
                const vector<OperatorID> &preferred = parent_preferred[j];
                helpful_membership[j] =
                    find(preferred.begin(), preferred.end(), op_id) != preferred.end();
            }
            insert_successor(
                i + 1, succ_state.get_id(), succ_node.get_g(), succ_h, helpful_membership);
        }
    }

    ++sweep_count;
    return IN_PROGRESS;
}

}
