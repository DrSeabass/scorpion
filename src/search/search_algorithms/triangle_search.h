#ifndef SEARCH_ALGORITHMS_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace triangle_search {

class TriangleSearch : public SearchAlgorithm {
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

    const int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;

    std::shared_ptr<Evaluator> eval;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::deque<OpenList> open_lists;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    bool has_non_empty_lists() const;
    void extend_open_lists(int num_lists);
    void trim_empty_lists();
    void update_incumbent(const State &goal_state);
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g, int &h_out);
    void insert_into_open_list(int list_index, const OpenEntry &entry);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    TriangleSearch(
        const std::shared_ptr<Evaluator> &eval,
        int slope,
        bool reopen_closed,
        bool anytime,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~TriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
