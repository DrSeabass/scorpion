#ifndef SEARCH_ALGORITHMS_RATCHET_BOOSTED_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_RATCHET_BOOSTED_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace ratchet_boosted_triangle_search {

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
  1a,2c) with the ratchet_triangle_search slope-adjustment mechanism ported
  on as-is and always on ("must have" -- no option to disable it). The
  cascade bound each step is max_active_layer + slope, where slope is a
  persistent state variable doubled or halved at the end of every step based
  on that step's heuristic-trend balance (strictly more informed than
  uninformed layer-transitions -> double; otherwise halve, floor 1). This
  trend signal is a single step-scoped counter over every expansion in the
  step regardless of which guidance list served it -- orthogonal to the
  per-list progress-credit budgets below, which track each list's own trend
  independently.

  At credit_boost=0 (default) list selection is bit-identical to
  boosted_triangle_search (and, with a single evaluator, to
  ratchet_triangle_search): the only behavioral difference from
  boosted_triangle_search is that slope now evolves instead of staying
  fixed. See ratchet_boosted_sweep_triangle_search for the once-per-dive
  sibling.

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
class RatchetBoostedTriangleSearch : public SearchAlgorithm {
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

    // Ratchet slope: persistent, doubled/halved at the end of every step
    // (see step()). Not a tunable knob -- the mechanism is always on and
    // this is its evolving state, ported as-is from ratchet_triangle_search.
    int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    const Schedule schedule;
    // ICAPS-27 progress credit (axis 1a, see icaps-27-plan.md), LAMA-style
    // token budget: an informed transition (a list's expansion h improves
    // on that same list's own previous expansion h at that layer) grants
    // it `credit_boost` tokens; every expansion it serves spends one token
    // (unclamped, can go negative). Selection serves the highest-budget
    // list (guidance heuristic, or the pruner if present) at each layer
    // boundary, ties broken by the schedule round-robin index.
    //
    // credit_boost == 0 (default) makes the whole mechanism inert: no
    // tokens are earned or spent, and selection short-circuits straight to
    // the schedule round-robin (see step()) -- an exact reduction, which is
    // what keeps the default configuration's list selection bit-identical
    // to boosted_triangle_search (slope still ratchets regardless).
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
    // round_robin_served. Never called at credit_boost == 0.
    int select_credit_served(
        const std::vector<HeuristicProgress> &budgets,
        int round_robin_served) const;
    // Applies one expansion's earn (if informed) and spend to hp.
    void record_expansion_credit(HeuristicProgress &hp, int h) const;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    RatchetBoostedTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int initial_slope,
        bool reopen_closed,
        bool anytime,
        Schedule schedule,
        int credit_boost,
        bool guide_by_pruning,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~RatchetBoostedTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
