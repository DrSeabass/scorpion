#ifndef SEARCH_ALGORITHMS_MULTI_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_MULTI_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace multi_triangle_search {

/*
  Direction A (multi-heuristic Triangle), rich variant. Each depth layer
  carries N parallel ranked open lists, one per inadmissible guidance
  heuristic in `evals`. A successor is evaluated by all N heuristics and
  inserted into all N lists at its layer, so the N lists are N orderings of
  the identical live frontier set -- they differ only in priority order and
  in transient stale copies. Duplicate detection stays global (one closed
  list); the within-layer drain discards stale copies per list. The optional
  admissible `pruning_heuristic` remains the single bound-pruner across all
  lists, unchanged from vanilla triangle.

  Scheduling (this commit): per-sweep round-robin. One heuristic list owns
  the entire cascade dive in a step; the served index rotates between steps
  (served = step_count % N), preserving the coherence of a single dive. With
  N == 1 the algorithm reduces to vanilla triangle.
*/
class MultiTriangleSearch : public SearchAlgorithm {
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

    std::vector<std::shared_ptr<Evaluator>> evals;
    const int num_lists;
    std::shared_ptr<Evaluator> pruning_heuristic;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::deque<LayerLists> open_lists;
    int max_active_layer = -1;
    // Per-sweep round-robin counter: the served list index is step_count % N.
    int step_count = 0;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void extend_open_lists(int num_layers);
    bool layer_empty(int layer) const;
    void recompute_max_active_layer();
    void update_incumbent(const State &goal_state);
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g,
        std::vector<int> &h_out, bool is_new_evaluation);
    void insert_successor(
        int layer, StateID id, int g, const std::vector<int> &hs);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    MultiTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int slope,
        bool reopen_closed,
        bool anytime,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~MultiTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
