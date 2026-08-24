#ifndef SEARCH_ALGORITHMS_LAZY_RATCHET_BOOSTED_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_LAZY_RATCHET_BOOSTED_TRIANGLE_SEARCH_H

#include "../evaluator.h"
#include "../open_list.h"
#include "../open_list_factory.h"
#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

class PruningMethod;

namespace lazy_ratchet_boosted_triangle_search {

enum class Schedule {
    SWEEP,
    POP
};

/*
  ICAPS-27 lazy ratchet-boosted triangle search (see
  icaps-27-lazy-eval-design.md and icaps-27-lazy-eval-implementation-prompt.md):
  lazy_boosted_triangle_search (this family's per-layer-reselection base)
  with the ratchet_triangle_search slope-adjustment mechanism ported on
  as-is and always on, exactly mirroring ratchet_boosted_triangle_search.
  The cascade bound each step is `layers.size() bound reached via slope`
  new layers, where slope is a persistent state variable doubled or halved
  at the end of every step based on that step's heuristic-trend balance
  (strictly more informed than uninformed layer-transitions -> double;
  otherwise halve, floor 1). This trend signal is a single step-scoped
  counter over every expansion in the step regardless of which guidance
  list served it -- orthogonal to the per-list progress-credit budgets,
  which track each list's own trend independently.

  Same lazy mechanics as lazy_boosted_triangle_search throughout: parent-h
  ranking, real evaluation (all N guidance heuristics) only at pop time,
  edges (not evaluated triples) as open-list entries so the credit signal
  always uses a freshly-computed real h, and preferred-operator sets
  computed once at pop time with no cache. See that file's header comment
  for the full rationale; this file only documents the ratchet diff.

  At num_lists == 1 and preferred_evals == [] (any credit_boost, any
  schedule), this evolves slope exactly as ratchet_triangle_search does,
  but there is no pre-existing *lazy* single-heuristic ratchet file to
  reduce to bit-for-bit -- see the implementation prompt's step 6 notes for
  how this file's correctness is instead verified (mechanism-level checks,
  not a bit-identical target).

  Optional pruner queue: see lazy_boosted_triangle_search.h for the full
  description (mirrored here as-is) of pruning_heuristic/guide_by_pruning,
  kept fully lazy -- evaluated only at pop time, alongside the guidance
  heuristics, never per generated successor. pruning_heuristic unset
  (default) is an exact no-op reduction to the pre-existing behavior.
*/
class LazyRatchetBoostedTriangleSearch : public SearchAlgorithm {
    enum class ExpansionOutcome {
        SKIPPED,
        EXPANDED,
        SOLVED
    };

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

    // Ratchet slope: persistent, doubled/halved at the end of every step
    // (see step()). Not a tunable knob -- the mechanism is always on and
    // this is its evolving state, ported as-is from ratchet_triangle_search.
    int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    const Schedule schedule;
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
    int select_credit_served(const Layer &layer, int round_robin_served) const;
    int select_round_robin_served(const Layer &layer, int round_robin_served) const;
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
    LazyRatchetBoostedTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int initial_slope,
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
    virtual ~LazyRatchetBoostedTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
