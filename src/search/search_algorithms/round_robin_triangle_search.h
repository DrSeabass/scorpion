#ifndef SEARCH_ALGORITHMS_ROUND_ROBIN_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_ROUND_ROBIN_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"
#include "../per_state_information.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class EvaluationContext;

namespace round_robin_triangle_search {

enum class Schedule { SWEEP, DEPTH };

/*
  Eager, fixed-slope, multi-heuristic triangle search for first-solution
  satisficing planning. Each depth layer contains one ordering of the shared
  frontier per evaluator and owns an independent round-robin cursor. A layer's
  cursor advances only when that layer actually supplies a live node for
  expansion; empty queues and queues exhausted by stale-entry removal do not
  advance it.
*/
class RoundRobinTriangleSearch : public SearchAlgorithm {
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

    using OpenList =
        std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryCompare>;

    struct Layer {
        std::vector<OpenList> lists;
        int next_served = 0;

        explicit Layer(int num_lists) : lists(num_lists) {}
    };

    const int slope;
    const bool reopen_closed_nodes;
    const Schedule schedule;
    std::vector<std::shared_ptr<Evaluator>> evals;
    const int num_lists;
    std::vector<std::shared_ptr<Evaluator>> preferred_evals;
    const int num_preferred;
    const int total_lists;
    std::vector<Evaluator *> path_dependent_evaluators;
    PerStateInformation<std::vector<OperatorID>> preferred_op_cache;

    std::deque<Layer> layers;
    int max_active_layer = -1;
    int sweep_count = 0;

    int guidance_index(int evaluator_index) const;
    int preferred_index(int evaluator_index) const;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void extend_layers(int num_layers);
    bool layer_empty(int layer) const;
    void recompute_max_active_layer();
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g,
        std::vector<int> &h_out, bool is_new_evaluation);
    void insert_successor(
        int layer, StateID id, int g, const std::vector<int> &hs,
        bool preferred);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    RoundRobinTriangleSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int slope,
        bool reopen_closed,
        Schedule schedule,
        const std::vector<std::shared_ptr<Evaluator>> &preferred_evals,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);

    virtual ~RoundRobinTriangleSearch() = default;
    virtual void print_statistics() const override;
};

}

#endif
