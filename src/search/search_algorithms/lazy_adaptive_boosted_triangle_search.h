#ifndef SEARCH_ALGORITHMS_LAZY_ADAPTIVE_BOOSTED_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_LAZY_ADAPTIVE_BOOSTED_TRIANGLE_SEARCH_H

#include "../evaluator.h"
#include "../open_list.h"
#include "../open_list_factory.h"
#include "../search_algorithm.h"

#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

class PruningMethod;

namespace lazy_adaptive_boosted_triangle_search {

enum class Schedule {
    SWEEP,
    POP
};

/*
  ICAPS-27 lazy adaptive-boosted triangle search (see
  icaps-27-lazy-eval-design.md and icaps-27-lazy-eval-implementation-prompt.md):
  lazy_boosted_triangle_search (this family's per-layer-reselection base)
  with the adaptive_triangle_search per-step depth-budget mechanism ported
  on as-is and always on, mirroring adaptive_boosted_triangle_search. The
  cascade runs until it would need to instantiate a new frontier layer it
  can't afford: a persistent depth_budget (starting at 1, floored at 1 each
  step) pays one unit per new frontier layer; an informed layer-transition
  (h decreases relative to the previous expansion this step) refunds one
  unit, an uninformed one debits one. The first frontier extension of a
  step is always free.

  Same lazy mechanics as lazy_boosted_triangle_search throughout: parent-h
  ranking, real evaluation only at pop time, edges as open-list entries,
  preferred-operator sets computed once at pop time with no cache.

  Structural note specific to this file (see the implementation prompt's
  step 6 notes for the full discussion): the eager sibling's budget gate
  works by peeking at the top of a priority_queue (non-destructive) before
  deciding whether to pay for a new frontier layer, leaving a
  budget-rejected entry physically in place for the next step to retry.
  EdgeOpenList (this family's open-list type) only exposes destructive
  remove_min() -- there is no peek. A single-slot "pending edge" buffer
  (pending_predecessor_id/pending_operator_id/pending_list_index/
  pending_absolute_depth) stands in for that non-destructive peek: at most
  one edge can ever be budget-rejected at a time (the cascade halts
  immediately when the gate fails), so one slot suffices. Because the edge
  is physically removed from its EdgeOpenList while pending,
  has_non_empty_lists()/trim_empty_layers() must not mistake its now-empty
  list for a truly exhausted layer -- see layer_empty()'s absolute-depth
  parameter.

  Optional pruner queue: see lazy_boosted_triangle_search.h for the full
  description (mirrored here as-is) of pruning_heuristic/guide_by_pruning,
  kept fully lazy -- evaluated only at pop time, alongside the guidance
  heuristics, never per generated successor. pruning_heuristic unset
  (default) is an exact no-op reduction to the pre-existing behavior.
*/
class LazyAdaptiveBoostedTriangleSearch : public SearchAlgorithm {
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
    // Adaptive per-step depth budget, ported as-is from
    // adaptive_triangle_search. Persistent across steps; reset to
    // max(1, remaining) at the top of every step.
    int depth_budget = 1;

    struct LayerDiagnostics {
        long long removals = 0;
        long long precheck_stale = 0;
        long long evaluated_skips = 0;
        long long goals = 0;
        long long expansions = 0;
    };
    std::map<int, LayerDiagnostics> layer_diagnostics;
    std::ofstream diagnostic_file;
    long long diagnostic_steps = 0;
    long long diagnostic_layers_considered = 0;
    long long diagnostic_nonempty_layers_considered = 0;
    int diagnostic_max_step_width = 0;

    void write_diagnostic_snapshot();

    // Single-slot pending-edge buffer -- see class comment above. "No
    // pending edge" <=> pending_operator_id == OperatorID::no_operator.
    StateID pending_predecessor_id = StateID::no_state;
    OperatorID pending_operator_id = OperatorID::no_operator;
    int pending_list_index = -1;
    int pending_absolute_depth = -1;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    bool has_non_empty_lists() const;
    // absolute_depth identifies the layer for pending-edge purposes (see
    // class comment): a list reported empty by the underlying EdgeOpenList
    // is still treated as non-empty if the pending edge belongs to this
    // exact (absolute_depth, list) pair.
    bool layer_empty(const Layer &layer, int absolute_depth) const;
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
    LazyAdaptiveBoostedTriangleSearch(
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
    virtual ~LazyAdaptiveBoostedTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
