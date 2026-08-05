#ifndef SEARCH_ALGORITHMS_ADAPTIVE_BOOSTED_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_ADAPTIVE_BOOSTED_TRIANGLE_SEARCH_H

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

namespace adaptive_boosted_triangle_search {

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

/*
  ICAPS-27 step 5 (see icaps-27-plan.md): boosted_triangle_search (combo
  1a,2c) with the adaptive_triangle_search per-step depth-budget mechanism
  ported on as-is and always on ("must have" -- no option to disable it).
  Unlike ratchet_boosted_triangle_search (a fixed-size cascade cap that
  itself evolves), this mechanism drops the cascade cap entirely: the
  cascade runs until it would need to instantiate a new frontier layer it
  can't afford. A persistent depth_budget (starting at 1, floored at 1 each
  step) pays one unit per new frontier layer; an informed layer-transition
  (h decreases relative to the previous expansion this step) refunds one
  unit, an uninformed one debits one (the original symmetric +1/-1 rule,
  hardwired -- adaptive_triangle_search's non_progress_penalty tunable is
  not exposed here, per the "no new knobs on a must-have mechanism" ground
  rule). The first frontier extension of a step is always free, so every
  step makes forward progress even with a depleted budget. This trend
  signal is a single step-scoped counter over every expansion in the step
  regardless of which guidance list served it -- orthogonal to the
  per-list progress-credit budgets below, which track each list's own
  trend independently.

  At credit_boost=0 (default) list selection is bit-identical to
  boosted_triangle_search (and, with a single evaluator, to
  adaptive_triangle_search): the only behavioral difference from
  boosted_triangle_search is that the cascade depth is now
  budget-gated instead of slope-capped. See
  adaptive_boosted_sweep_triangle_search for the once-per-dive sibling.

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
  which of its operators are preferred. This is entirely orthogonal to the
  adaptive depth-budget mechanism above: helpful lists affect which list
  is served, the depth budget affects how many layers the cascade covers.

  Because a helpful list can be empty at a layer that still has live
  content elsewhere, "the served list is empty" no longer implies "the
  layer is empty" once any preferred_evals are configured. Both selection
  helpers (select_round_robin_served, select_credit_served) therefore pick
  the best-ranked list among the layer's currently non-empty lists, rather
  than assuming the first pick is representative of the whole layer. At
  preferred_evals=[], no list can ever be selectively empty like this
  (every guidance/pruner list holds the same live content), so both
  helpers reduce to their pre-step-6 behavior exactly.

  Scheduling: per-sweep round-robin. One heuristic list owns the entire
  cascade dive in a step; the served index rotates between steps (served =
  sweep_count % N). With N == 1 the algorithm reduces to vanilla triangle.
*/
class AdaptiveBoostedTriangleSearch : public SearchAlgorithm {
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
    // ICAPS-27 progress credit (axis 1a, see icaps-27-plan.md), LAMA-style
    // token budget: an informed transition (a list's expansion h improves
    // on that same list's own previous expansion h at that layer) grants
    // it `credit_boost` tokens; every expansion it serves spends one token
    // (unclamped, can go negative). Selection serves the highest-budget
    // list (guidance heuristic, helpful list, or the pruner if present)
    // at each layer boundary, ties broken by the schedule round-robin
    // index.
    //
    // credit_boost == 0 (default) makes the whole mechanism inert: no
    // tokens are earned or spent, and selection short-circuits straight to
    // the schedule round-robin (see step()) -- an exact reduction, which is
    // what keeps the default configuration's list selection bit-identical
    // to boosted_triangle_search.
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
    // together, so there is no separate container to keep in sync.
    struct Layer {
        LayerLists lists;
        std::vector<HeuristicProgress> progress;
        explicit Layer(int total_lists) : lists(total_lists), progress(total_lists) {}
    };
    std::deque<Layer> layers;
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
    // Picks the served list index among the layer's currently non-empty
    // lists (guidance heuristics, helpful lists, and the optional pruner
    // list all compete on equal footing) by highest budget, ties broken by
    // round_robin_served. Never called at credit_boost == 0.
    int select_credit_served(const Layer &layer, int round_robin_served) const;
    // Picks round_robin_served if its list is non-empty; otherwise walks
    // the round-robin order (wrapping) to the next non-empty list. Reduces
    // to round_robin_served exactly whenever preferred_evals is empty
    // (every guidance/pruner list then shares identical live content, so
    // round_robin_served's own list can only be empty when every list is).
    int select_round_robin_served(const Layer &layer, int round_robin_served) const;
    // Applies one expansion's earn (if informed) and spend to hp.
    void record_expansion_credit(HeuristicProgress &hp, int h) const;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    AdaptiveBoostedTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        bool reopen_closed,
        bool anytime,
        Schedule schedule,
        int credit_boost,
        const std::vector<std::shared_ptr<Evaluator>> &preferred_evals,
        bool guide_by_pruning,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~AdaptiveBoostedTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
