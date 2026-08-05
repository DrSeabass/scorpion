#ifndef SEARCH_ALGORITHMS_ADAPTIVE_BOOSTED_SWEEP_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_ADAPTIVE_BOOSTED_SWEEP_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace adaptive_boosted_sweep_triangle_search {

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
  ICAPS-27 step 5 (see icaps-27-plan.md): boosted_sweep_triangle_search
  (combos 1a,2d / 1b,2d) with the adaptive_triangle_search per-step
  depth-budget mechanism ported on as-is and always on ("must have" -- no
  option to disable it). Unlike ratchet_boosted_sweep_triangle_search (a
  fixed-size cascade cap that itself evolves), this mechanism drops the
  cascade cap entirely: the cascade runs until it would need to
  instantiate a new frontier layer it can't afford. A persistent
  depth_budget (starting at 1, floored at 1 each step) pays one unit per
  new frontier layer; an informed layer-transition (h decreases relative
  to the previous expansion this step) refunds one unit, an uninformed
  one debits one (the original symmetric +1/-1 rule, hardwired --
  adaptive_triangle_search's non_progress_penalty tunable is not exposed
  here). The first frontier extension of a step is always free. This trend
  signal is a single step-scoped counter over every expansion in the step
  regardless of which guidance list served it -- orthogonal to the
  progress-credit budgets below (per-layer or global), which decide the
  dive's served list once, before the cascade loop.

  At credit_boost=0 (default) list selection is bit-identical to
  boosted_sweep_triangle_search (and, with a single evaluator, to
  adaptive_triangle_search): the only behavioral difference from
  boosted_sweep_triangle_search is that the cascade depth is now
  budget-gated instead of slope-capped. See adaptive_boosted_triangle_search
  for the per-layer reselection sibling.

  Each depth layer carries N parallel ranked open lists, one per
  inadmissible guidance heuristic in `evals`. A successor is evaluated by
  all N heuristics and inserted into all N lists at its layer, so the N
  lists are N orderings of the identical live frontier set -- they differ
  only in priority order and in transient stale copies. Duplicate
  detection stays global (one closed list); the within-layer drain
  discards stale copies per list. The optional admissible
  `pruning_heuristic` remains the single bound-pruner across all lists,
  unchanged from vanilla triangle.
*/
class AdaptiveBoostedSweepTriangleSearch : public SearchAlgorithm {
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
    // configuration's list selection bit-identical to
    // boosted_sweep_triangle_search.
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
    // no separate container to keep in sync. The budgets here are read (and
    // updated) only when credit_scope == PER_LAYER.
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
    // Adaptive per-step depth budget (see class comment above), ported
    // as-is from adaptive_triangle_search. Persistent across steps; reset
    // to max(1, remaining) at the top of every step.
    int depth_budget = 1;

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
    // round_robin_served. Used directly on global_progress for
    // credit_scope == GLOBAL. Never called at credit_boost == 0.
    int select_credit_served(
        const std::vector<HeuristicProgress> &budgets,
        int round_robin_served) const;
    // Same idea, but for credit_scope == PER_LAYER: sums budgets across
    // every currently active layer first, for the single once-per-dive
    // decision.
    int select_sweep_served(int round_robin_served) const;
    // Applies one expansion's earn (if informed) and spend to hp.
    void record_expansion_credit(HeuristicProgress &hp, int h) const;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    AdaptiveBoostedSweepTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        bool reopen_closed,
        bool anytime,
        Schedule schedule,
        CreditScope credit_scope,
        int credit_boost,
        bool guide_by_pruning,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~AdaptiveBoostedSweepTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
