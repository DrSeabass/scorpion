#ifndef SEARCH_ALGORITHMS_LAZY_BOOSTED_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_LAZY_BOOSTED_TRIANGLE_SEARCH_H

#include "../evaluator.h"
#include "../open_list.h"
#include "../open_list_factory.h"
#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

class PruningMethod;

namespace lazy_boosted_triangle_search {

// How the per-layer cascade picks which of the N guidance lists to pop.
// See boosted_triangle_search.h for the eager original; duplicated here
// (not shared) matching this branch's own sibling-file precedent.
enum class Schedule {
    SWEEP,
    POP
};

/*
  ICAPS-27 lazy boosted triangle search (see icaps-27-lazy-eval-design.md
  and icaps-27-lazy-eval-implementation-prompt.md): lazy sibling of
  boosted_triangle_search. Successors are ranked by their *parent's*
  already-known h (reused for free via EvaluationContext's copy
  constructor, exactly as lazy_triangle_search and FD's own LazySearch do
  it) and are only evaluated for real -- with all N guidance heuristics --
  when popped for expansion. Every successor is inserted into all
  num_lists lists (one per evaluator in `evals`), each ranked by that
  evaluator's own parent-h.

  Progress-credit boosting (axis 1a, per-layer -- see icaps-27-plan.md) and
  selection (axis 2c, per-layer reselection, matching
  boosted_triangle_search's own combo) are both live. credit_boost == 0
  (default) makes the mechanism inert -- selection reduces to plain
  round-robin (schedule SWEEP/POP), which reduces further to serving list 0
  whenever num_lists == 1.

  Because entries in this file's open lists are *edges*
  (predecessor_id, operator_id), not {id, h, g} triples, there is no
  stored-h field that could go stale the way a naive lazy retrofit of the
  eager entry type would risk (see icaps-27-lazy-eval-design.md Q3): the
  credit signal always uses the state's freshly-computed real h from the
  same process_candidate call that is expanding it, by construction, not a
  cached value read back off an entry.

  ICAPS-27 step 6 / implementation step 4 (helpful actions): each evaluator
  named in `preferred_evals` (a subset of `evals`, matched by identity)
  gets one additional "helpful" list per layer, ranked by that same
  evaluator's parent-h (sharing the guidance list's own ranking, no extra
  evaluation) but populated only with successors reached via one of that
  evaluator's own preferred operators on the parent. Unlike the eager
  sibling, there is no PerStateInformation cache here: per
  icaps-27-lazy-eval-design.md Q4, a state's own preferred-operator sets
  are computed once, in process_candidate, at the moment it is popped and
  evaluated for real -- which is also the only moment they are ever
  needed, to decide helpful-list membership for the successors generated
  in that same call. preferred_evals=[] (default) adds no helpful lists
  and is an exact no-op reduction to the pre-step-4 behavior above.

  No pruner queue -- pruning_heuristic is never in use in the configs
  motivating this work (see icaps-27-lazy-eval-design.md Q2), so unlike
  the eager family this lazy family never grows a guide_by_pruning option.

  At num_lists == 1 and preferred_evals == [] (any credit_boost, any
  schedule), this must reduce exactly (identical expansion counts and
  plan) to lazy_triangle(eval=evals[0], slope=slope).
*/
class LazyBoostedTriangleSearch : public SearchAlgorithm {
    enum class ExpansionOutcome {
        SKIPPED,
        EXPANDED,
        SOLVED
    };

    // Per-heuristic budget state for the token economy (see
    // boosted_triangle_search.h for the original description).
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
    // ICAPS-27 progress credit (axis 1a, per-layer): see
    // boosted_triangle_search.h's own field for the full description.
    // credit_boost == 0 (default) is an exact no-op reduction to plain
    // round-robin selection.
    const int credit_boost;

    std::vector<std::shared_ptr<Evaluator>> evals;
    const int num_lists;
    // ICAPS-27 step 6: subset of `evals` (matched by pointer identity,
    // validated in the constructor) whose preferred operators each get a
    // paired helpful list.
    std::vector<std::shared_ptr<Evaluator>> preferred_evals;
    const int num_preferred;
    // For helpful list j (0 <= j < num_preferred), the index in
    // [0, num_lists) of the guidance evaluator it shares its
    // h-value/ranking with.
    std::vector<int> preferred_source_index;
    // Lists per layer: num_lists guidance lists, then num_preferred
    // helpful lists.
    const int total_lists;
    std::vector<std::shared_ptr<OpenListFactory>> open_list_factories;
    std::shared_ptr<PruningMethod> pruning_method;

    std::vector<Evaluator *> path_dependent_evaluators;

    std::deque<Layer> layers;
    bool root_pending;
    // Per-sweep round-robin counter: the served list index is sweep_count % N.
    int sweep_count = 0;
    // Per-pop round-robin counter: advances per expansion (POP schedule only).
    int pop_count = 0;
    // Absolute depth of layers[0], for debug logging of drained layers'
    // final budgets against a stable depth label.
    int depth_offset = 0;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    bool has_non_empty_lists() const;
    bool layer_empty(const Layer &layer) const;
    Layer create_layer() const;
    void extend_layers(int num_layers);
    void trim_empty_layers();
    void update_incumbent(const State &goal_state);
    // Picks the served list index among the layer's currently non-empty
    // lists by highest budget, ties broken by round_robin_served. Never
    // called at credit_boost == 0.
    int select_credit_served(const Layer &layer, int round_robin_served) const;
    // Picks round_robin_served if its list is non-empty; otherwise walks
    // the round-robin order (wrapping) to the next non-empty list.
    int select_round_robin_served(const Layer &layer, int round_robin_served) const;
    // Applies one expansion's earn (if informed) and spend to hp, using the
    // real (just-computed) h -- never a value read back off an entry.
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
    LazyBoostedTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int slope,
        bool reopen_closed,
        bool anytime,
        Schedule schedule,
        int credit_boost,
        const std::vector<std::shared_ptr<Evaluator>> &preferred_evals,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~LazyBoostedTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
