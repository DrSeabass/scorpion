#ifndef SEARCH_ALGORITHMS_BOOSTED_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_BOOSTED_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace boosted_triangle_search {

// How the per-step cascade picks which of the N guidance lists to pop.
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

// ICAPS-27 axis 2 (see icaps-27-plan.md): how often the credit-driven
// heuristic selector (step 4) will reselect the active guidance list.
// Orthogonal to Schedule (which governs raw round-robin popping) and to
// slope adjustment (the adaptive_triangle/ratchet_triangle per-step budget
// mechanism ported in step 5) -- neither of those is affected by this
// option. Unused until step 4 wires credit into selection; parsed and
// logged only for now.
enum class SelectionGranularity {
    // Reselect the active guidance list at each layer boundary.
    PER_LAYER,
    // Lock the active guidance list for the whole dive; reselect only
    // between sweeps.
    PER_SWEEP
};

// ICAPS-27 axis 1 (see icaps-27-plan.md): progress-credit granularity.
// Orthogonal to SelectionGranularity, except that GLOBAL + PER_LAYER
// (combo 1b,2c) is excluded by the plan as not meaningful and rejected at
// construction.
enum class CreditScope {
    // One independent token budget per depth layer (combos 1a,2c / 1a,2d).
    PER_LAYER,
    // One token budget shared across the whole search (combo 1b,2d only).
    GLOBAL
};

/*
  ICAPS-27 boosted triangle search: sibling of multi_triangle_search
  growing progress-credit boosting, heuristic-selection granularity, slope
  adjustment, and helpful-action filtering (see icaps-27-plan.md). At
  credit_boost=0 (default) this is bit-identical to multi_triangle_search
  regardless of selection_granularity or credit_scope; a nonzero
  credit_boost engages a LAMA-style per-list token budget -- tracked per
  layer or globally (credit_scope) -- that reselects the served list per
  layer boundary or once per dive (selection_granularity). Combo 1b,2c
  (credit_scope=global, selection_granularity=per_layer) is rejected at
  construction.

  Each depth layer carries N parallel ranked open lists, one per
  inadmissible guidance heuristic in `evals`. A successor is evaluated by
  all N heuristics and inserted into all N lists at its layer, so the N
  lists are N orderings of the identical live frontier set -- they differ
  only in priority order and in transient stale copies. Duplicate
  detection stays global (one closed list); the within-layer drain
  discards stale copies per list. The optional admissible
  `pruning_heuristic` remains the single bound-pruner across all lists,
  unchanged from vanilla triangle.

  Scheduling (this commit): per-sweep round-robin. One heuristic list owns
  the entire cascade dive in a step; the served index rotates between steps
  (served = sweep_count % N), preserving the coherence of a single dive. With
  N == 1 the algorithm reduces to vanilla triangle.
*/
class BoostedTriangleSearch : public SearchAlgorithm {
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
    // ICAPS-27 axis 2: how often the credit-driven selector below reselects
    // the served list -- PER_LAYER at each layer boundary (combo 1a,2c),
    // PER_SWEEP once per dive from credit summed across active layers
    // (combo 1a,2d). Both reduce to the plain schedule round-robin whenever
    // credit_boost == 0 (see below), which is the only way the default
    // PER_SWEEP stays bit-identical to multi_triangle_search.
    const SelectionGranularity selection_granularity;
    // ICAPS-27 axis 1: whether progress credit (below) is tracked per layer
    // (default, combos 1a,2c / 1a,2d) or as a single budget shared across
    // the whole search (combo 1b,2d).
    const CreditScope credit_scope;
    // ICAPS-27 progress credit (axis 1a), LAMA-style token budget: an
    // informed transition (a list's expansion h improves on that same
    // list's own previous expansion h) grants it `credit_boost` tokens;
    // every expansion it serves spends one token (unclamped, can go
    // negative). Selection serves the highest-budget list (guidance
    // heuristic, or the pruner if present), ties broken by the schedule
    // round-robin index.
    //
    // credit_boost == 0 (default) makes the whole mechanism inert: no
    // tokens are earned or spent, and selection short-circuits straight to
    // the schedule round-robin (see step()) -- an exact reduction,
    // regardless of selection_granularity. This is what keeps the default
    // configuration (selection_granularity=per_sweep, credit_boost=0)
    // bit-identical to multi_triangle_search.
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
    // Whether the admissible pruner contributes an extra ranked list.
    const bool use_pruner_queue;
    // Lists per layer: num_lists guidance lists plus the optional pruner list.
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
    // per list (the pruner list, if any, competes on equal footing). Both
    // halves are always the same size and live or die together, so there is
    // no separate container to keep in sync.
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

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void extend_layers(int num_layers);
    bool layer_empty(int layer) const;
    void recompute_max_active_layer();
    void update_incumbent(const State &goal_state);
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g,
        std::vector<int> &h_out, bool is_new_evaluation);
    void insert_successor(
        int layer, StateID id, int g, const std::vector<int> &hs);
    // Picks the served list index among all total_lists lists (guidance
    // heuristics plus the optional pruner list, which competes on equal
    // footing) given their current budgets, ties broken by
    // round_robin_served.
    int select_credit_served(
        const std::vector<HeuristicProgress> &budgets,
        int round_robin_served) const;
    // ICAPS-27 axis 1a x 2d: same idea as select_credit_served, but the
    // budgets are summed across every currently active layer first, for a
    // single once-per-dive decision.
    int select_sweep_served(int round_robin_served) const;
    // Applies one expansion's earn (if informed) and spend to hp.
    void record_expansion_credit(HeuristicProgress &hp, int h) const;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    BoostedTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int slope,
        bool reopen_closed,
        bool anytime,
        Schedule schedule,
        SelectionGranularity selection_granularity,
        CreditScope credit_scope,
        int credit_boost,
        bool guide_by_pruning,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~BoostedTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
