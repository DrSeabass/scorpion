#include "lazy_boosted_triangle_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../open_lists/best_first_open_list.h"
#include "../plan_manager.h"
#include "../pruning_method.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"

#include "../utils/logging.h"
#include "../utils/system.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <set>

using namespace std;

namespace lazy_boosted_triangle_search {

namespace {
// Store the lazy ranking key at insertion. Reading a batch boundary must not
// evaluate successor states or change keys when a predecessor is reopened.
class EqualEvaluationOpenList : public EdgeOpenList {
    shared_ptr<Evaluator> evaluator;
    map<pair<int, int>, deque<EdgeOpenListEntry>> buckets;

    void do_insertion(EvaluationContext &context, const EdgeOpenListEntry &entry) override {
        buckets[{context.get_evaluator_value(evaluator.get()), context.get_g_value()}]
            .push_back(entry);
    }

public:
    explicit EqualEvaluationOpenList(const shared_ptr<Evaluator> &evaluator)
        : evaluator(evaluator) {}

    pair<int, int> min_key() const {
        assert(!empty());
        return buckets.begin()->first;
    }

    EdgeOpenListEntry remove_min() override {
        assert(!empty());
        auto it = buckets.begin();
        EdgeOpenListEntry entry = it->second.front();
        it->second.pop_front();
        if (it->second.empty())
            buckets.erase(it);
        return entry;
    }

    bool empty() const override { return buckets.empty(); }
    void clear() override { buckets.clear(); }
    void get_path_dependent_evaluators(set<Evaluator *> &evals) override {
        evaluator->get_path_dependent_evaluators(evals);
    }
    bool is_dead_end(EvaluationContext &context) const override {
        return context.is_evaluator_value_infinite(evaluator.get());
    }
    bool is_reliable_dead_end(EvaluationContext &context) const override {
        return evaluator->dead_ends_are_reliable() && is_dead_end(context);
    }
};
}

LazyBoostedTriangleSearch::LazyBoostedTriangleSearch(
    const vector<shared_ptr<Evaluator>> &evals,
    int slope,
    bool reopen_closed,
    bool anytime,
    Schedule schedule,
    int credit_boost,
    bool union_preferred,
    bool skip_empty_on_sweep,
    const vector<shared_ptr<Evaluator>> &preferred_evals,
    bool guide_by_pruning,
    const shared_ptr<Evaluator> &pruning_heuristic,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity, bool global_preferred,
    int k, bool expand_equal, bool random_start, int random_seed, int window)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      slope(slope),
      reopen_closed_nodes(reopen_closed),
      anytime_search(anytime),
      schedule(schedule),
      credit_boost(credit_boost),
      union_preferred(union_preferred),
      skip_empty_on_sweep(skip_empty_on_sweep),
      global_preferred(global_preferred),
      k(k),
      expand_equal(expand_equal),
      sweep_start(random_start, random_seed, window),
      evals(evals),
      num_lists(static_cast<int>(evals.size())),
      preferred_evals(preferred_evals),
      num_preferred(static_cast<int>(preferred_evals.size())),
      total_lists(
          static_cast<int>(evals.size()) +
          (global_preferred ? 1 : static_cast<int>(preferred_evals.size())) +
          (guide_by_pruning && pruning_heuristic != nullptr ? 1 : 0)),
      pruning_method(pruning),
      guide_by_pruning(guide_by_pruning),
      use_pruner_queue(guide_by_pruning && pruning_heuristic != nullptr),
      pruning_heuristic(pruning_heuristic),
      root_pending(true) {
    if (k < 1 || (expand_equal && k != 1)) {
        cerr << "k must be positive; expand_equal=true requires k=1." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (global_preferred && sweep_start.enabled()) {
        cerr << "Custom sweep starts require depth-stratified queues." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (global_preferred && (schedule != Schedule::SWEEP ||
                            preferred_evals.empty() || !union_preferred ||
                            !skip_empty_on_sweep || credit_boost != 0)) {
        cerr << "global_preferred requires unboosted union-preferred sweep scheduling "
                "and nonempty preferred_evals." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if ((k != 1 || expand_equal) &&
        (schedule == Schedule::POP || credit_boost != 0 || !skip_empty_on_sweep)) {
        cerr << "Batch expansion requires lazy_multi_triangle sweep or depth scheduling." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (slope <= 0) {
        cerr << "LazyBoostedTriangleSearch: slope must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    if (evals.empty()) {
        cerr << "LazyBoostedTriangleSearch: at least one guidance evaluator is required." << endl;
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
            cerr << "LazyBoostedTriangleSearch: every preferred_evals entry must "
                    "also appear in evals." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        preferred_source_index.push_back(source);
    }
    if (union_preferred && num_preferred != 0 && num_preferred != num_lists) {
        cerr << "LazyBoostedTriangleSearch: union-preferred mode requires "
                "preferred_evals to contain every guidance evaluator."
             << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
    open_list_factories.resize(total_lists);
    for (int k = 0; k < num_lists; ++k) {
        open_list_factories[guidance_queue_index(k)] =
            make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(
                evals[k], false);
    }
    for (int j = 0; j < num_preferred && !global_preferred; ++j) {
        int source = preferred_source_index[j];
        open_list_factories[preferred_queue_index(j)] =
            make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(
                evals[source], false);
    }
    if (global_preferred) {
        // An unused placeholder keeps layer queue indices stable.
        open_list_factories[num_lists] = open_list_factories[0];
    }
    if (use_pruner_queue) {
        open_list_factories[total_lists - 1] =
            make_shared<standard_scalar_open_list::BestFirstOpenListFactory>(
                pruning_heuristic, false);
    }
}

int LazyBoostedTriangleSearch::guidance_queue_index(int evaluator_index) const {
    if (global_preferred || !union_preferred || num_preferred == 0)
        return evaluator_index;
    return 2 * evaluator_index;
}

int LazyBoostedTriangleSearch::preferred_queue_index(int preferred_index) const {
    if (global_preferred || !union_preferred || num_preferred == 0)
        return num_lists + preferred_index;
    return 2 * preferred_source_index[preferred_index] + 1;
}

void LazyBoostedTriangleSearch::initialize() {
    log << "Conducting lazy multi-heuristic triangle search with slope " << slope
        << ", " << num_lists << " guidance heuristic(s)"
        << ", schedule = "
        << (schedule == Schedule::SWEEP ? "sweep" :
            schedule == Schedule::POP ? "pop" : "depth")
        << ", credit_boost = " << credit_boost
        << ", " << (global_preferred ? 1 : num_preferred)
        << " preferred-operator (helpful) list(s)"
        << ", global_preferred = " << global_preferred
        << ", k = " << k << ", expand_equal = " << expand_equal
        << ", guide_by_pruning = " << use_pruner_queue
        << " (" << (global_preferred ? num_lists : total_lists) << " list(s)/layer)"
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

    layers.clear();
    layers.push_back(create_layer());
    root_pending = true;

    pruning_method->initialize(task);
}

void LazyBoostedTriangleSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();

    // ICAPS-27 progress credit (axis 1a): final per-heuristic budgets for
    // any layers still live when the search ended.
    if (log.is_at_least_debug()) {
        for (size_t i = 0; i < layers.size(); ++i) {
            log << "Lazy boosted triangle layer " << (depth_offset + static_cast<int>(i))
                << " final budgets:";
            for (int k = 0; k < total_lists; ++k)
                log << " " << layers[i].progress[k].budget;
            log << endl;
        }
    }
}

void LazyBoostedTriangleSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    // Report f-value progress against the primary guidance heuristic.
    int value = eval_context.get_evaluator_value_or_infinity(evals[0].get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

bool LazyBoostedTriangleSearch::has_non_empty_lists() const {
    if (root_pending)
        return true;
    for (const Layer &layer : layers) {
        if (!layer_empty(layer))
            return true;
    }
    return false;
}

bool LazyBoostedTriangleSearch::layer_empty(const Layer &layer) const {
    if (layer.global_entries > 0)
        return false;
    for (const unique_ptr<EdgeOpenList> &list : layer.lists) {
        if (!list->empty())
            return false;
    }
    return true;
}

LazyBoostedTriangleSearch::Layer LazyBoostedTriangleSearch::create_layer() const {
    vector<unique_ptr<EdgeOpenList>> lists(total_lists);
    if (expand_equal) {
        for (int j = 0; j < num_lists; ++j)
            lists[guidance_queue_index(j)] = make_unique<EqualEvaluationOpenList>(evals[j]);
        if (!global_preferred) {
            for (int j = 0; j < num_preferred; ++j)
                lists[preferred_queue_index(j)] =
                    make_unique<EqualEvaluationOpenList>(evals[preferred_source_index[j]]);
        }
        if (global_preferred)
            lists[num_lists] = open_list_factories[num_lists]->create_edge_open_list();
        if (use_pruner_queue)
            lists[total_lists - 1] = make_unique<EqualEvaluationOpenList>(pruning_heuristic);
    } else {
        for (int j = 0; j < total_lists; ++j)
            lists[j] = open_list_factories[j]->create_edge_open_list();
    }
    return Layer(std::move(lists));
}

void LazyBoostedTriangleSearch::extend_layers(int num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(create_layer());
    }
}

void LazyBoostedTriangleSearch::trim_empty_layers() {
    while (!layers.empty() && layer_empty(layers.front())) {
        layers.pop_front();
        ++depth_offset;
    }
    while (!layers.empty() && layer_empty(layers.back())) {
        layers.pop_back();
    }
}

int LazyBoostedTriangleSearch::select_credit_served(
    const Layer &layer, int round_robin_served) const {
    // ICAPS-27 step 6: helpful lists can be selectively empty (unlike
    // guidance lists, which always hold identical live content), so only
    // lists with live entries at this layer are eligible.
    int best = -1;
    for (int k = 0; k < total_lists; ++k) {
        if (layer.lists[k]->empty())
            continue;
        if (best == -1 || layer.progress[k].budget > layer.progress[best].budget)
            best = k;
    }
    if (best == -1)
        return round_robin_served;
    if (!layer.lists[round_robin_served]->empty() &&
        layer.progress[round_robin_served].budget == layer.progress[best].budget)
        return round_robin_served;
    return best;
}

int LazyBoostedTriangleSearch::select_round_robin_served(
    const Layer &layer, int round_robin_served) const {
    // ICAPS-27 step 6: walk the round-robin order (wrapping) until a
    // non-empty list is found. At preferred_evals=[] this always returns
    // round_robin_served itself on the very first check (every guidance
    // list shares identical live content), an exact reduction to the
    // pre-step-6 behavior.
    for (int offset = 0; offset < total_lists; ++offset) {
        int k = (round_robin_served + offset) % total_lists;
        if (!layer.lists[k]->empty())
            return k;
    }
    return round_robin_served;
}

void LazyBoostedTriangleSearch::record_expansion_credit(
    HeuristicProgress &hp, int h) const {
    if (hp.have_last_h && h < hp.last_h)
        hp.budget += credit_boost;
    hp.have_last_h = true;
    hp.last_h = h;
    --hp.budget;
}

void LazyBoostedTriangleSearch::update_incumbent(const State &goal_state) {
    Plan candidate_plan =
        search_space.trace_path(task_proxy, successor_generator, goal_state);
    int candidate_cost = calculate_plan_cost(candidate_plan, task_proxy);

    if (!found_solution() || candidate_cost < bound) {
        set_plan(candidate_plan);
        bound = candidate_cost;
        log << "LazyBoostedTriangleSearch: improved incumbent with cost " << candidate_cost << endl;
        if (anytime_search) {
            plan_manager.save_plan(candidate_plan, task_proxy, true);
        }
    }
}

LazyBoostedTriangleSearch::ExpansionOutcome LazyBoostedTriangleSearch::process_candidate(
    const State &state,
    StateID predecessor_id,
    OperatorID operator_id,
    int g,
    int real_g,
    int source_layer_index,
    bool is_root,
    vector<int> &h_out) {
    SearchNode node = search_space.get_node(state);

    if (!reopen_closed_nodes && !node.is_new())
        return ExpansionOutcome::SKIPPED;

    bool reopen = reopen_closed_nodes && node.is_closed() && !node.is_dead_end() && (g < node.get_g());

    if (!(node.is_new() || reopen))
        return ExpansionOutcome::SKIPPED;

    if (!is_root) {
        assert(predecessor_id != StateID::no_state);
        assert(operator_id != OperatorID::no_operator);
        if (!path_dependent_evaluators.empty()) {
            State parent_state = state_registry.lookup_state(predecessor_id);
            for (Evaluator *evaluator : path_dependent_evaluators) {
                evaluator->notify_state_transition(parent_state, operator_id, state);
            }
        }
    }

    statistics.inc_evaluated_states();
    // ICAPS-27 step 6: request preferred operators on this context whenever
    // helpful lists are in use, so any evaluator queried below computes
    // them as a side effect of its normal h-value call.
    EvaluationContext eval_context(state, g, true, &statistics, num_preferred > 0);

    // Evaluate all N guidance heuristics from a single context now that the
    // state is actually being popped for expansion -- laziness pays this
    // cost only for states that reach this point, never for states that
    // are merely generated. A state is dead iff any reliable heuristic
    // proves it so (multi-source dead-end, same rule as the eager sibling).
    bool is_dead_end = false;
    h_out.assign(total_lists, 0);
    for (int k = 0; k < num_lists; ++k) {
        int h = eval_context.get_evaluator_value_or_infinity(evals[k].get());
        if (h == EvaluationResult::INFTY && evals[k]->dead_ends_are_reliable())
            is_dead_end = true;
        h_out[guidance_queue_index(k)] = h;
    }
    // ICAPS-27 step 6: helpful lists share their h-value with their source
    // guidance list -- no extra evaluation, just a copy at the right index.
    for (int j = 0; j < num_preferred && !global_preferred; ++j)
        h_out[preferred_queue_index(j)] =
            h_out[guidance_queue_index(preferred_source_index[j])];
    if (is_dead_end) {
        node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return ExpansionOutcome::SKIPPED;
    }

    // The admissible pruning_heuristic stays lazy exactly like the guidance
    // heuristics: evaluated here, once, only for a state that is actually
    // being popped for expansion -- never per generated successor. A state
    // exceeding the current bound is skipped without being marked a dead
    // end (it may still be reachable more cheaply via a different
    // predecessor later, e.g. after reopening); one the heuristic reliably
    // proves unsolvable is marked dead end, exactly like the guidance
    // heuristics above.
    if (pruning_heuristic) {
        int prune_h = eval_context.get_evaluator_value_or_infinity(pruning_heuristic.get());
        if (prune_h == EvaluationResult::INFTY) {
            if (pruning_heuristic->dead_ends_are_reliable()) {
                node.mark_as_dead_end();
                statistics.inc_dead_ends();
            }
            return ExpansionOutcome::SKIPPED;
        }
        if (bound != numeric_limits<int>::max() && g + prune_h >= bound)
            return ExpansionOutcome::SKIPPED;
        // The pruner list's own real h, for the credit signal if it is
        // ever served -- no second evaluation, this is the same value just
        // computed above for the f-prune check.
        if (use_pruner_queue)
            h_out[total_lists - 1] = prune_h;
    }

    // ICAPS-27 step 6: this state's own preferred-operator sets, computed
    // once, right now, at the only moment they are ever needed -- to decide
    // helpful-list membership for the successors generated later in this
    // same call. No cache: per icaps-27-lazy-eval-design.md Q4, there is no
    // earlier "generation time" separate from this one evaluation point to
    // cache against.
    vector<vector<OperatorID>> preferred_ops(num_preferred);
    for (int j = 0; j < num_preferred; ++j)
        preferred_ops[j] = eval_context.get_preferred_operators(preferred_evals[j].get());

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
        return anytime_search ? ExpansionOutcome::SKIPPED : ExpansionOutcome::SOLVED;
    }

    node.close();
    statistics.inc_expanded();

    if (search_progress.check_progress(eval_context)) {
        statistics.print_checkpoint_line(g);
    }

    while (static_cast<int>(layers.size()) <= source_layer_index + 1) {
        layers.push_back(create_layer());
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

        // Rank the successor by the parent's already-known h, reusing the
        // parent's EvaluationContext cache (the copy constructor keeps the
        // parent's state, so any evaluator queried here returns the
        // parent's value for free) -- exactly lazy_triangle_search's and
        // FD's LazySearch's own idiom, generalized to num_lists evaluators.
        int succ_g = g + get_adjusted_cost(op);
        EvaluationContext succ_eval_context(eval_context, succ_g, false, nullptr);
        // Guidance lists (and the optional pruner list, ranked by the same
        // parent-h proxy -- pruning_heuristic stays lazy just like the
        // guidance heuristics, see the class comment) always receive every
        // successor.
        for (int k = 0; k < num_lists; ++k) {
            layers[source_layer_index + 1].lists[guidance_queue_index(k)]->insert(
                succ_eval_context, make_pair(state.get_id(), op_id));
        }
        if (use_pruner_queue) {
            layers[source_layer_index + 1].lists[total_lists - 1]->insert(
                succ_eval_context, make_pair(state.get_id(), op_id));
        }
        // Helpful lists either use their evaluator's own preferred set (the
        // historical boosted-triangle behavior) or LAMA's union of all
        // preferred evaluators (the lazy_multi_triangle behavior).
        for (int j = 0; j < num_preferred; ++j) {
            bool is_preferred = false;
            if (union_preferred) {
                for (const vector<OperatorID> &preferred : preferred_ops) {
                    if (find(preferred.begin(), preferred.end(), op_id) !=
                        preferred.end()) {
                        is_preferred = true;
                        break;
                    }
                }
            } else {
                const vector<OperatorID> &preferred = preferred_ops[j];
                is_preferred =
                    find(preferred.begin(), preferred.end(), op_id) != preferred.end();
            }
            if (is_preferred && global_preferred) {
                preferred_queue.push_back({make_pair(state.get_id(), op_id),
                                           depth_offset + source_layer_index + 1});
                ++layers[source_layer_index + 1].global_entries;
                break; // Insert once even if several evaluators prefer this edge.
            }
            if (is_preferred) {
                layers[source_layer_index + 1].lists[preferred_queue_index(j)]->insert(
                    succ_eval_context, make_pair(state.get_id(), op_id));
            }
        }
    }

    return ExpansionOutcome::EXPANDED;
}

SearchStatus LazyBoostedTriangleSearch::step() {
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

    // Find the deepest queued layer before adding the slope allowance.
    // In lazy search the frontier consists of unevaluated successor edges.
    int deepest_layer = static_cast<int>(layers.size()) - 1;
    while (deepest_layer > 0 && layer_empty(layers[deepest_layer]))
        --deepest_layer;
    const int first_layer = root_pending ? 0 : sweep_start.choose(depth_offset, deepest_layer);
    if (sweep_start.enabled() && log.is_at_least_debug()) {
        log << "Triangle sweep start: depth=" << depth_offset + first_layer
            << " front=" << depth_offset + deepest_layer
            << " offset=" << depth_offset << endl;
    }
    extend_layers(slope);

    // SWEEP: one guidance list owns the entire cascade dive this step; the
    // served index rotates between steps. POP: the served index advances
    // per expansion (below), so successive expansions down the dive
    // alternate heuristics.
    const int sweep_served = sweep_count % total_lists;

    // Snapshot the budget: a FIFO expansion may append deeper layers.
    const int num_layers = static_cast<int>(layers.size());
    for (int i = first_layer; i < num_layers - 1; ++i) {
        if (i == 0 && root_pending) {
            root_pending = false;
            State initial_state = state_registry.get_initial_state();
            vector<int> h_out;
            ExpansionOutcome outcome = process_candidate(
                initial_state,
                StateID::no_state,
                OperatorID::no_operator,
                0, 0, i, true, h_out);
            if (outcome == ExpansionOutcome::SOLVED)
                return SOLVED;
            continue;
        }

        if (global_preferred && sweep_served == num_lists && preferred_queue.empty())
            break;

        const int round_robin_served =
            schedule == Schedule::SWEEP
                ? sweep_served
                : schedule == Schedule::POP
                    ? pop_count % total_lists
                    : layers[i].next_served;
        // ICAPS-27 axis 1a x 2c (see icaps-27-plan.md): reselect the served
        // list at this layer boundary from its own budgets. credit_boost ==
        // 0 falls straight through to the round-robin selector -- no
        // credit consulted, no budgets touched below.
        int served =
            (credit_boost == 0)
                ? ((schedule == Schedule::SWEEP && skip_empty_on_sweep)
                       ? round_robin_served
                       : select_round_robin_served(
                             layers[i], round_robin_served))
                : select_credit_served(layers[i], round_robin_served);
        bool expanded = false;
        int expanded_count = 0;
        pair<int, int> batch_key;
        const int max_lists_to_try =
            schedule == Schedule::DEPTH ? total_lists : 1;
        for (int lists_tried = 0;
             lists_tried < max_lists_to_try && !expanded;
             ++lists_tried) {
            EdgeOpenList &list = *layers[i].lists[served];
            const bool serve_global = global_preferred && served == num_lists;
            while ((serve_global ? !preferred_queue.empty() : !list.empty()) &&
                   (serve_global ? expanded_count < 1 :
                    (expand_equal || expanded_count < k))) {
                pair<int, int> candidate_key;
                if (expand_equal && !serve_global) {
                    candidate_key = static_cast<EqualEvaluationOpenList &>(list).min_key();
                    if (expanded_count > 0 && candidate_key != batch_key)
                        break;
                }
                int expansion_layer = i;
                EdgeOpenListEntry next{StateID::no_state, OperatorID::no_operator};
                if (serve_global) {
                    PreferredEntry entry = preferred_queue.front();
                    preferred_queue.pop_front();
                    next = entry.edge;
                    expansion_layer = entry.depth - depth_offset;
                    assert(expansion_layer >= 0 &&
                           expansion_layer < static_cast<int>(layers.size()));
                    --layers[expansion_layer].global_entries;
                } else {
                    next = list.remove_min();
                }

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
                // Scorpion SearchNode lacks get_real_g(); get_g() equals
                // real_g for NORMAL cost type.
                int real_g = predecessor_node.get_g() + op.get_cost();
                State state = state_registry.get_successor_state(predecessor, op);

                vector<int> h_out;
                ExpansionOutcome outcome = process_candidate(
                    state, predecessor_id, operator_id, g, real_g, expansion_layer, false,
                    h_out);
                if (outcome == ExpansionOutcome::SOLVED)
                    return SOLVED;
                if (outcome == ExpansionOutcome::EXPANDED) {
                    expanded = true;
                    if (expanded_count == 0)
                        batch_key = candidate_key;
                    ++expanded_count;
                    if (log.is_at_least_debug()) {
                        log << "Triangle expansion: sweep=" << sweep_count
                            << " queue=" << (serve_global ? "preferred" : "guidance")
                            << " depth=" << expansion_layer + depth_offset
                            << " list=" << served;
                        if (expand_equal && !serve_global)
                            log << " h=" << candidate_key.first << " g=" << candidate_key.second;
                        log << endl;
                    }
                    if (credit_boost != 0) {
                        record_expansion_credit(
                            layers[i].progress[served], h_out[served]);
                    }
                    ++pop_count;
                }
            }
            // For per-depth scheduling, a raw-nonempty queue can prove to be
            // entirely stale. Continue cyclically to another queue in the
            // same visit. The persistent cursor changes only if a queue
            // actually supplies a live expansion.
            if (!expanded && schedule == Schedule::DEPTH) {
                served = select_round_robin_served(
                    layers[i], (served + 1) % total_lists);
            }
        }
        // One queue owns the whole batch; rotate only after a live batch.
        if (schedule == Schedule::DEPTH && expanded)
            layers[i].next_served = (served + 1) % total_lists;
    }

    trim_empty_layers();
    ++sweep_count;
    return IN_PROGRESS;
}

}
