#ifndef SEARCH_ALGORITHMS_RATCHET_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_RATCHET_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <fstream>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace ratchet_triangle_search {

class RatchetTriangleSearch : public SearchAlgorithm {
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

    int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    // Direction B (relaxed cascade start-depth): when true, begin each step's
    // cascade slope-1 layers below the shallowest active layer instead of at
    // the root. Off => start index stays 0 => bit-identical to vanilla ratchet.
    const bool lift_floor;
    // When true, append one CSV row per step (expansions,slope) to the
    // hardcoded file ratchet_triangle_slope.csv, tracing the ratcheted slope.
    // Off => no file is opened.
    const bool log_slope;

    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<Evaluator> pruning_heuristic;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::deque<OpenList> open_lists;
    int max_active_layer = -1;
    // Absolute depth of open_lists[0]: incremented whenever an empty front
    // layer is popped, so reported depths stay absolute despite front-draining.
    int depth_offset = 0;

    // Open when log_slope is set; one row per step.
    std::ofstream slope_log_file;
    // Open when log_slope is set; one row index per incumbent improvement,
    // marking the end-of-step log element at which a solution was found.
    std::ofstream slope_solution_file;
    // Count of rows written to slope_log_file so far; equals the index of the
    // next (upcoming) step-end row.
    int log_rows_written = 0;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void extend_open_lists(int num_lists);
    void recompute_max_active_layer();
    void update_incumbent(const State &goal_state);
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g, int &h_out,
        bool is_new_evaluation);
    void insert_into_open_list(int list_index, const OpenEntry &entry);
    // Absolute depths of the shallowest / deepest non-empty open list (-1 if
    // all empty), used for the mindepth/maxdepth logging columns.
    void current_depth_range(int &mind, int &maxd) const;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    RatchetTriangleSearch(
        const std::shared_ptr<Evaluator> &eval,
        int initial_slope,
        bool reopen_closed,
        bool anytime,
        bool lift_floor,
        bool log_slope,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~RatchetTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
