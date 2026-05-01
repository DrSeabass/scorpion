#ifndef SEARCH_ALGORITHMS_LAZY_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_LAZY_TRIANGLE_SEARCH_H

#include "../evaluator.h"
#include "../open_list.h"
#include "../open_list_factory.h"
#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

class PruningMethod;

namespace lazy_triangle_search {

class LazyTriangleSearch : public SearchAlgorithm {
    enum class ExpansionOutcome {
        SKIPPED,
        EXPANDED,
        SOLVED
    };

    const int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;

    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<OpenListFactory> open_list_factory;
    std::shared_ptr<PruningMethod> pruning_method;

    std::vector<Evaluator *> path_dependent_evaluators;

    std::deque<std::unique_ptr<EdgeOpenList>> open_lists;
    bool root_pending;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    bool has_non_empty_lists() const;
    std::unique_ptr<EdgeOpenList> create_open_list() const;
    void extend_open_lists(int num_lists);
    void trim_empty_lists();
    void update_incumbent(const State &goal_state);

    ExpansionOutcome process_candidate(
        const State &state,
        StateID predecessor_id,
        OperatorID operator_id,
        int g,
        int real_g,
        int source_list_index,
        bool is_root);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    LazyTriangleSearch(
        const std::shared_ptr<Evaluator> &eval,
        int slope,
        bool reopen_closed,
        bool anytime,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~LazyTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
