#ifndef SEARCH_ALGORITHMS_BOOSTED_SWEEP_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_BOOSTED_SWEEP_TRIANGLE_SEARCH_H

#include "../operator_id.h"
#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class EvaluationContext;
class PruningMethod;

namespace boosted_sweep_triangle_search {

// How the per-step cascade picks which of the N guidance lists to pop when
// credit_boost == 0 (the token budget below is inert and this is the whole
// story). Also the round-robin tie-break/fallback once credit is engaged.
enum class Schedule {
    // Per-sweep round-robin: one guidance list owns the entire cascade dive
    // this step; the served index rotates between steps (served =
    // sweep_count % N). Preserves the coherence of a single dive.
    SWEEP,
    // Per-pop round-robin: the served index advances per expansion, so
    // successive expansions down a dive alternate heuristics. The literal
    // multi-heuristic-alternation analog at expansion granularity.
    POP
};

// ICAPS-27 axis 1 (see icaps-27-plan.md): progress-credit granularity.
enum class CreditScope {
    // Budgets tracked per depth layer, summed across every layer currently
    // active for the once-per-dive decision (combo 1a,2d).
    PER_LAYER,
    // One token budget shared across the whole search (combo 1b,2d).
    GLOBAL
};

/*
  ICAPS-27 boosted-sweep triangle search (see icaps-27-plan.md): the
  once-per-dive sibling of boosted_triangle_search -- the served list is
  decided once per step(), before the cascade loop, and locked for the
  whole dive (combos 1a,2d / 1b,2d; credit_scope picks between them). At
  credit_boost=0 (default) this is bit-identical to multi_triangle_search
  regardless of credit_scope; a nonzero credit_boost engages a LAMA-style
  token budget -- summed across active layers (credit_scope=per_layer) or
  read directly from a single shared budget (credit_scope=global) -- that
  picks the dive's served list.

  Each depth layer carries N parallel ranked open lists, one per
  inadmissible guidance heuristic in `evals`. A successor is evaluated by
  all N heuristics and inserted into all N lists at its layer, so the N
  lists are N orderings of the identical live frontier set -- they differ
  only in priority order and in transient stale copies. Duplicate
  detection stays global (one closed list); the within-layer drain
  discards stale copies per list. The optional admissible
  `pruning_heuristic` remains the single bound-pruner across all lists,
  unchanged from vanilla triangle, and (like the guidance lists) receives
  every live successor.

  ICAPS-27 step 6 (helpful actions, see icaps-27-plan.md): each evaluator
  named in `preferred_evals` (a subset of `evals`, matched by identity) gets
  one additional "helpful" list per layer, ranked by that same evaluator's
  h but populated only with successors reached via one of that evaluator's
  own preferred operators on the parent -- unlike the guidance/pruner
  lists, a helpful list is a sparse subset of the layer's live content.
  preferred_evals=[] (default) adds no helpful lists and is an exact no-op
  reduction to today's behavior. Preferred-operator sets are computed once
  per state, when it is first evaluated as a successor (see
  evaluate_and_prepare_node), and persisted until that state is popped for
  expansion -- so no evaluator is ever run a second time just to learn
  which of its operators are preferred.

  Because a helpful list can be empty at a layer that still has live
  content elsewhere, the list locked in for the whole dive (locked_served)
  can turn up empty at some specific layer even though it was clearly
  eligible when the once-per-dive decision was made (or, at credit_boost
  == 0 with schedule=pop, the recomputed-per-layer round-robin candidate
  can likewise be momentarily empty). select_available_served() below
  handles this as a narrow escape hatch -- find the nearest non-empty list
  to the intended one at this specific layer -- without re-litigating the
  once-per-dive policy choice itself. At preferred_evals=[], no list can
  ever be selectively empty like this (every guidance/pruner list holds
  the same live content), so it reduces to the intended list exactly.
*/
class BoostedSweepTriangleSearch : public SearchAlgorithm {
    struct OpenEntry {
        StateID id;
        int h;
        int g;
    };

    struct OpenEntryCompare {
        bool operator()(const OpenEntry &lhs, const OpenEntry &rhs) const {
            if (lhs.h != rhs.h)
                return lhs.h > rhs.h;
            return lhs.g > rhs.g;
        }
    };

    using OpenList = std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryCompare>;
    // One layer holds N ranked lists, one per guidance heuristic.
    using LayerLists = std::vector<OpenList>;

    const int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    const Schedule schedule;
    // ICAPS-27 axis 1: whether progress credit (below) is tracked per layer
    // (combo 1a,2d) or as a single budget shared across the whole search
    // (combo 1b,2d).
    const CreditScope credit_scope;
    // ICAPS-27 progress credit, LAMA-style token budget: an informed
    // transition (a list's expansion h improves on that same list's own
    // previous expansion h) grants it `credit_boost` tokens; every
    // expansion it serves spends one token (unclamped, can go negative).
    // The served list for the whole dive is the highest-budget one, ties
    // broken by the schedule round-robin index.
    //
    // credit_boost == 0 (default) makes the whole mechanism inert: no
    // tokens are earned or spent, and selection short-circuits straight to
    // the schedule round-robin (see step()) -- an exact reduction,
    // regardless of credit_scope. This is what keeps the default
    // configuration bit-identical to multi_triangle_search.
    const int credit_boost;
    // When true (and a pruning_heuristic is set), the admissible heuristic
    // also gets its own ranked list at index num_lists, joining the
    // round-robin. The admissible h is already computed for the f-prune, so
    // this extra queue costs list memory, not evaluation -- but it forces the
    // prune eval to be unconditional (computed even before the first
    // incumbent) to seed the queue.
    const bool guide_by_pruning;

    std::vector<std::shared_ptr<Evaluator>> evals;
    // Number of inadmissible guidance heuristics.
    const int num_lists;
    // ICAPS-27 step 6 (helpful actions): subset of `evals` (matched by
    // pointer identity, validated in the constructor) whose preferred
    // operators each get a paired helpful list.
    std::vector<std::shared_ptr<Evaluator>> preferred_evals;
    // Number of helpful lists (== preferred_evals.size()).
    const int num_preferred;
    // For helpful list j (0 <= j < num_preferred), the index in [0,
    // num_lists) of the guidance evaluator it shares its h-value/ranking
    // with -- i.e. which evals[] slot preferred_evals[j] matches.
    std::vector<int> preferred_source_index;
    // Whether the admissible pruner contributes an extra ranked list.
    const bool use_pruner_queue;
    // Lists per layer: num_lists guidance lists, then num_preferred helpful
    // lists, then the optional pruner list.
    const int total_lists;
    std::shared_ptr<Evaluator> pruning_heuristic;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    // Per-heuristic budget state for the token economy described above.
    struct HeuristicProgress {
        bool have_last_h = false;
        int last_h = 0;
        int budget = 0;
    };
    // One depth layer: its total_lists ranked open lists, plus one budget
    // per list (the pruner list, if any, and every helpful list compete on
    // equal footing). Both halves are always the same size and live or die
    // together, so there is no separate container to keep in sync. The
    // budgets here are read (and updated) only when credit_scope ==
    // PER_LAYER.
    struct Layer {
        LayerLists lists;
        std::vector<HeuristicProgress> progress;
        explicit Layer(int total_lists) : lists(total_lists), progress(total_lists) {}
    };
    std::deque<Layer> layers;
    // ICAPS-27 axis 1b (global): one shared budget per list, used instead
    // of layers[i].progress when credit_scope == GLOBAL. Sized to
    // total_lists in initialize().
    std::vector<HeuristicProgress> global_progress;
    int max_active_layer = -1;
    // Per-sweep round-robin counter: the served list index is sweep_count % N.
    int sweep_count = 0;
    // Per-pop round-robin counter: advances per expansion (POP schedule only),
    // so the served list index is pop_count % N at each expansion.
    int pop_count = 0;
    // Absolute depth of layers[0], so a popped layer's final budgets can be
    // logged against a stable depth label.
    int depth_offset = 0;
    // ICAPS-27 step 6 (helpful actions): a state's preferred-operator sets
    // (one vector<OperatorID> per preferred_evals entry), cached once when
    // the state is first evaluated as a successor (see
    // evaluate_and_prepare_node) and consumed when it is later popped for
    // expansion (see step()). Empty (default-constructed) for any state
    // when num_preferred == 0, so this costs nothing when helpful actions
    // are unused.
    PerStateInformation<std::vector<std::vector<OperatorID>>> preferred_op_cache;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void extend_layers(int num_layers);
    bool layer_empty(int layer) const;
    void recompute_max_active_layer();
    void update_incumbent(const State &goal_state);
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g,
        std::vector<int> &h_out, bool is_new_evaluation);
    // ICAPS-27 step 6: extracts preferred operators for `state` from
    // `eval_context` (which must have been constructed with
    // calculate_preferred=true) for every preferred_evals entry, and caches
    // them for later consumption at expansion time. No-op when
    // num_preferred == 0.
    void cache_preferred_operators(
        const State &state, EvaluationContext &eval_context);
    void insert_successor(
        int layer, StateID id, int g, const std::vector<int> &hs,
        const std::vector<bool> &helpful_membership);
    // Picks the served list index among all total_lists lists (guidance
    // heuristics plus the optional pruner list, which competes on equal
    // footing) given their current budgets, ties broken by
    // round_robin_served. Used directly on global_progress for
    // credit_scope == GLOBAL. Never called at credit_boost == 0. This is a
    // once-per-dive policy decision, not scoped to any specific layer, so
    // it is not itself aware of per-layer emptiness -- see
    // select_available_served for how that is handled.
    int select_credit_served(
        const std::vector<HeuristicProgress> &budgets,
        int round_robin_served) const;
    // Same idea, but for credit_scope == PER_LAYER: sums budgets across
    // every currently active layer first, for the single once-per-dive
    // decision.
    int select_sweep_served(int round_robin_served) const;
    // ICAPS-27 step 6: the once-per-dive policy decision above (or the
    // per-layer round-robin candidate at credit_boost == 0) can turn out
    // to be empty at a specific layer once helpful lists exist. Walks the
    // round-robin order (wrapping) from `intended` until a non-empty list
    // at this layer is found, without re-litigating the policy decision
    // itself. Reduces to `intended` exactly whenever preferred_evals is
    // empty (every guidance/pruner list then shares identical live
    // content, so `intended`'s own list can only be empty when every list
    // is).
    int select_available_served(const Layer &layer, int intended) const;
    // Applies one expansion's earn (if informed) and spend to hp.
    void record_expansion_credit(HeuristicProgress &hp, int h) const;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    BoostedSweepTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int slope,
        bool reopen_closed,
        bool anytime,
        Schedule schedule,
        CreditScope credit_scope,
        int credit_boost,
        const std::vector<std::shared_ptr<Evaluator>> &preferred_evals,
        bool guide_by_pruning,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~BoostedSweepTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
