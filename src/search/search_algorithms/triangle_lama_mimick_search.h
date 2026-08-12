#ifndef SEARCH_ALGORITHMS_TRIANGLE_LAMA_MIMICK_SEARCH_H
#define SEARCH_ALGORITHMS_TRIANGLE_LAMA_MIMICK_SEARCH_H

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

namespace triangle_lama_mimick_search {

/*
  Triangle-LAMA-mimick search (see icaps-27-plan.md): boosted_sweep_triangle
  (schedule=sweep) with LAMA's own alternation-queue policy substituted for
  boosted_sweep_triangle's self-referential progress-credit mechanism.

  Every list -- each guidance heuristic's list, each helpful/preferred-only
  list (from `preferred_evals`), and the optional pruner list -- carries one
  persistent, global (not per-layer) priority counter, all starting at 0.
  This is exactly alternation_open_list::AlternationOpenList's `priorities`.
  The served list for a whole cascade dive is decided once per step(),
  before the cascade loop (this is the "sweep" part): the lowest-counter
  list wins, ties keep the lowest index -- AlternationOpenList::remove_min's
  selection rule. Every expansion the served list actually serves (per
  layer, honoring the empty-list fallback below) costs it one point. With
  no boosting this alone is round-robin over all lists, in index order.

  The boost, LAMA's actual mechanism (see alternation_open_list.cc,
  lazy_search.cc): whenever a newly generated successor's evaluation
  reports a new global-best value for any evaluator used for boosting
  (every Heuristic, unconditionally -- see Heuristic's Evaluator base
  constructor), every helpful/preferred-only list's counter drops by
  `boost_amount`, exactly like AlternationOpenList::boost_preferred(). The
  trigger is global (any evaluator's global minimum) and the reward always
  lands on the whole preferred-list category, never on whichever heuristic
  happened to cause the progress -- unlike boosted_sweep_triangle's
  credit_boost, which is self-referential per list and per layer.

  This is a deliberate translation, not a literal copy: real LAMA is lazy
  (heuristic evaluation deferred to when a node is popped for expansion),
  so it checks/rewards progress at expansion time, and its initial state's
  own evaluation never rewards (only later expansions do -- see
  LazySearch::step() vs. LazySearch::initialize()). This search's
  heuristics are eager (evaluated once, at generation time), so the check
  happens where the state is first evaluated -- see
  evaluate_and_prepare_node -- which is also never called for the initial
  state, preserving the same "no reward from the initial evaluation" rule
  by construction.

  preferred_evals=[] (default) leaves no helpful lists to ever boost, so
  this reduces exactly to round-robin over the guidance (and optional
  pruner) lists -- i.e. multi_triangle_search with schedule=sweep.
  boost_amount=0 is an equivalent way to disable boosting while still
  declaring helpful lists.

  Otherwise structurally identical to boosted_sweep_triangle: a deque of
  depth layers, each holding one ranked open list per guidance heuristic in
  `evals` (a successor is evaluated by all of them and inserted into all
  their lists at its layer -- duplicate detection stays global), plus one
  helpful list per `preferred_evals` entry (sparse, populated only via that
  evaluator's own preferred operator on the parent), plus an optional
  pruner list when guide_by_pruning is set. A locked-in list that is
  momentarily empty at a specific layer falls back to the nearest
  non-empty list (select_available_served) without re-litigating the
  once-per-dive choice.
*/
class TriangleLamaMimickSearch : public SearchAlgorithm {
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
    // One layer holds total_lists ranked lists: num_lists guidance lists,
    // then num_preferred helpful lists, then the optional pruner list.
    using Layer = std::vector<OpenList>;

    const int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    // LAMA's boost_preferred magnitude (see alternation_open_list.cc); 0
    // makes boosting inert without removing the helpful lists themselves.
    const int boost_amount;
    // When true (and a pruning_heuristic is set), the admissible heuristic
    // also gets its own ranked list at index total_lists-1, joining the
    // round-robin/boost pool as a plain (never-boosted) list.
    const bool guide_by_pruning;

    std::vector<std::shared_ptr<Evaluator>> evals;
    // Number of inadmissible guidance heuristics.
    const int num_lists;
    // Subset of `evals` (matched by pointer identity, validated in the
    // constructor) whose preferred operators each get a paired helpful
    // list -- LAMA's preferred-only queues, and the only lists
    // boost_amount ever touches.
    std::vector<std::shared_ptr<Evaluator>> preferred_evals;
    // Number of helpful lists (== preferred_evals.size()).
    const int num_preferred;
    // For helpful list j (0 <= j < num_preferred), the index in [0,
    // num_lists) of the guidance evaluator it shares its h-value/ranking
    // with.
    std::vector<int> preferred_source_index;
    // Whether the admissible pruner contributes an extra ranked list.
    const bool use_pruner_queue;
    const int total_lists;
    std::shared_ptr<Evaluator> pruning_heuristic;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::deque<Layer> layers;
    // AlternationOpenList-style priority counters, one per list, global
    // (not per depth layer) and persistent across the whole search.
    std::vector<int> priorities;
    int max_active_layer = -1;
    // Absolute depth of layers[0].
    int depth_offset = 0;
    // A state's preferred-operator sets (one vector<OperatorID> per
    // preferred_evals entry), cached once when the state is first
    // evaluated as a successor and consumed when it is later popped for
    // expansion. Empty for any state when num_preferred == 0.
    PerStateInformation<std::vector<std::vector<OperatorID>>> preferred_op_cache;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void extend_layers(int num_layers);
    bool layer_empty(int layer) const;
    void recompute_max_active_layer();
    void update_incumbent(const State &goal_state);
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g,
        std::vector<int> &h_out, bool is_new_evaluation);
    void cache_preferred_operators(
        const State &state, EvaluationContext &eval_context);
    void insert_successor(
        int layer, StateID id, int g, const std::vector<int> &hs,
        const std::vector<bool> &helpful_membership);
    // AlternationOpenList::remove_min's selection rule: lowest priority
    // counter wins, ties keep the lowest index. This is a once-per-dive
    // policy decision, not scoped to any specific layer -- see
    // select_available_served for how per-layer emptiness is handled.
    int select_served() const;
    // The once-per-dive pick above can turn out to be empty at a specific
    // layer once helpful lists exist. Walks the index order (wrapping)
    // from `intended` until a non-empty list at this layer is found,
    // without re-litigating the once-per-dive choice itself. Reduces to
    // `intended` exactly whenever preferred_evals is empty.
    int select_available_served(const Layer &layer, int intended) const;
    // AlternationOpenList::boost_preferred(): drop every helpful list's
    // priority by boost_amount. No-op when num_preferred == 0.
    void boost_preferred_lists();

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    TriangleLamaMimickSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int slope,
        bool reopen_closed,
        bool anytime,
        int boost_amount,
        const std::vector<std::shared_ptr<Evaluator>> &preferred_evals,
        bool guide_by_pruning,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~TriangleLamaMimickSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
