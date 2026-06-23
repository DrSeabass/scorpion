#ifndef SEARCH_ALGORITHMS_RECTANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_RECTANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace rectangle_search {

class RectangleSearch : public SearchAlgorithm {
    const int beam_width;
    const int aspect;
    const bool reopen_closed_nodes;
    const bool anytime_search;

    std::shared_ptr<Evaluator> eval;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::vector<std::deque<StateID>> open_lists;
    int depth;

    struct Candidate {
        StateID id;
        int eval_value;
        int g;
    };

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void update_incumbent(const State &goal_state);
    bool handle_goal(const State &state);
    bool select_and_expand(int list_index);
    void insert_into_open_list(int list_index, const Candidate &candidate);
    void extend_open_lists(int num_lists);
    void trim_empty_lists();
    bool has_non_empty_lists() const;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    RectangleSearch(
        const std::shared_ptr<Evaluator> &eval,
        int beam_width,
        int aspect,
        bool reopen_closed,
        bool anytime,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~RectangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
