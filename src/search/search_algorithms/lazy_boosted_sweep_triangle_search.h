#ifndef SEARCH_ALGORITHMS_LAZY_BOOSTED_SWEEP_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_LAZY_BOOSTED_SWEEP_TRIANGLE_SEARCH_H

#include "../evaluator.h"
#include "../open_list.h"
#include "../open_list_factory.h"
#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

class PruningMethod;

namespace lazy_boosted_sweep_triangle_search {

// How the per-step cascade picks which of the N guidance lists to pop when
// credit_boost == 0. Also the round-robin tie-break/fallback once credit is
// engaged. See boosted_sweep_triangle_search.h for the eager original;
// duplicated here (not shared) matching this branch's own sibling-file
// precedent.
enum class Schedule {
    SWEEP,
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
  ICAPS-27 lazy boosted-sweep triangle search (see
  icaps-27-lazy-eval-design.md and icaps-27-lazy-eval-implementation-prompt.md):
  the once-per-dive sibling of lazy_boosted_triangle_search -- the served
  list is decided once per step(), before the cascade loop, and locked for
  the whole dive (combos 1a,2d / 1b,2d; credit_scope picks between them).
  Same lazy mechanics as the per-layer sibling: successors are ranked by
  their parent's already-known h and only evaluated for real (with all N
  guidance heuristics) when popped for expansion; open-list entries are
  edges, so there is no stored-h field the credit signal could read a stale
  value from -- it always uses the state's freshly-computed real h from the
  same process_candidate call that is expanding it.

  ICAPS-27 step 6 (helpful actions): identical mechanism to the per-layer
  sibling -- preferred-operator sets are computed once, in process_candidate,
  at the moment a state is popped and evaluated for real, and used
  immediately to decide helpful-list membership for the successors
  generated in that same call. No PerStateInformation cache.

  Because a helpful list can be empty at a layer that still has live
  content elsewhere, the list locked in for the whole dive (locked_served)
  can turn up empty at some specific layer even though it was clearly
  eligible when the once-per-dive decision was made. select_available_served
  below handles this as a narrow escape hatch -- find the nearest non-empty
  list to the intended one at this specific layer -- without re-litigating
  the once-per-dive policy choice itself. At preferred_evals=[], no list can
  ever be selectively empty like this, so it reduces to the intended list
  exactly.

  Optional pruner queue: see lazy_boosted_triangle_search.h for the full
  description (mirrored here as-is) of pruning_heuristic/guide_by_pruning,
  kept fully lazy -- evaluated only at pop time, alongside the guidance
  heuristics, never per generated successor. pruning_heuristic unset
  (default) is an exact no-op reduction to the pre-existing behavior.

  At num_lists == 1 and preferred_evals == [] (any credit_boost, any
  schedule, any credit_scope), this must reduce exactly (identical
  expansion counts and plan) to lazy_triangle(eval=evals[0], slope=slope).
*/
class LazyBoostedSweepTriangleSearch : public SearchAlgorithm {
    enum class ExpansionOutcome {
        SKIPPED,
        EXPANDED,
        SOLVED
    };

    // Per-heuristic budget state for the token economy.
    struct HeuristicProgress {
        bool have_last_h = false;
        int last_h = 0;
        int budget = 0;
    };

    struct Layer {
        std::vector<std::unique_ptr<EdgeOpenList>> lists;
        std::vector<HeuristicProgress> progress;
        explicit Layer(std::vector<std::unique_ptr<EdgeOpenList>> &&lists_)
            : lists(std::move(lists_)), progress(lists.size()) {}
    };

    const int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    const Schedule schedule;
    // ICAPS-27 axis 1: whether progress credit is tracked per layer (combo
    // 1a,2d) or as a single budget shared across the whole search (combo
    // 1b,2d).
    const CreditScope credit_scope;
    const int credit_boost;

    std::vector<std::shared_ptr<Evaluator>> evals;
    const int num_lists;
    std::vector<std::shared_ptr<Evaluator>> preferred_evals;
    const int num_preferred;
    std::vector<int> preferred_source_index;
    // Lists per layer: num_lists guidance lists, then num_preferred
    // helpful lists, then the optional pruner list.
    const int total_lists;
    std::vector<std::shared_ptr<OpenListFactory>> open_list_factories;
    std::shared_ptr<PruningMethod> pruning_method;
    const bool guide_by_pruning;
    const bool use_pruner_queue;
    std::shared_ptr<Evaluator> pruning_heuristic;

    std::vector<Evaluator *> path_dependent_evaluators;

    std::deque<Layer> layers;
    // ICAPS-27 axis 1b (global): one shared budget per list, used instead
    // of layers[i].progress when credit_scope == GLOBAL. Sized to
    // total_lists in initialize().
    std::vector<HeuristicProgress> global_progress;
    bool root_pending;
    int sweep_count = 0;
    int pop_count = 0;
    int depth_offset = 0;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    bool has_non_empty_lists() const;
    bool layer_empty(const Layer &layer) const;
    Layer create_layer() const;
    void extend_layers(int num_layers);
    void trim_empty_layers();
    void update_incumbent(const State &goal_state);
    // Picks the served list index among all total_lists lists given their
    // current budgets, ties broken by round_robin_served. Used directly on
    // global_progress for credit_scope == GLOBAL. Never called at
    // credit_boost == 0. A once-per-dive policy decision, not scoped to any
    // specific layer -- see select_available_served for per-layer emptiness.
    int select_credit_served(
        const std::vector<HeuristicProgress> &budgets, int round_robin_served) const;
    // Same idea, but for credit_scope == PER_LAYER: sums budgets across
    // every currently active layer first, for the single once-per-dive
    // decision.
    int select_sweep_served(int round_robin_served) const;
    // ICAPS-27 step 6: the once-per-dive policy decision above (or the
    // per-layer round-robin candidate at credit_boost == 0) can turn out to
    // be empty at a specific layer once helpful lists exist. Walks the
    // round-robin order (wrapping) from `intended` until a non-empty list
    // at this layer is found, without re-litigating the policy decision.
    int select_available_served(const Layer &layer, int intended) const;
    void record_expansion_credit(HeuristicProgress &hp, int h) const;

    ExpansionOutcome process_candidate(
        const State &state,
        StateID predecessor_id,
        OperatorID operator_id,
        int g,
        int real_g,
        int source_layer_index,
        bool is_root,
        std::vector<int> &h_out);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    LazyBoostedSweepTriangleSearch(
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
    virtual ~LazyBoostedSweepTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
