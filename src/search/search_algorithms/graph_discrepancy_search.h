#ifndef SEARCH_ALGORITHMS_GRAPH_DISCREPANCY_SEARCH_H
#define SEARCH_ALGORITHMS_GRAPH_DISCREPANCY_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace graph_discrepancy_search {

class GraphDiscrepancySearch : public SearchAlgorithm {
    enum class DiscrepancyMode {
        BINARY,
        CHILD_RANK,
        H_GAP
    };

    struct OpenEntry {
        StateID id;
        int total_discrepancy;
        int g;
        unsigned long long insertion_id;
    };

    struct OpenEntryCompare {
        bool operator()(const OpenEntry &lhs, const OpenEntry &rhs) const {
            if (lhs.total_discrepancy != rhs.total_discrepancy)
                return lhs.total_discrepancy > rhs.total_discrepancy;
            if (lhs.g != rhs.g)
                return lhs.g > rhs.g;
            return lhs.insertion_id > rhs.insertion_id;
        }
    };

    struct SuccessorCandidate {
        OperatorID op_id;
        State succ_state;
        SearchNode succ_node;
        int succ_g;
        int rank_h;
    };

    const bool reopen_closed_nodes;
    const bool anytime_search;
    const DiscrepancyMode discrepancy_mode;

    std::shared_ptr<Evaluator> discrepancy_eval;
    std::shared_ptr<Evaluator> prune_eval;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryCompare>
        open;
    PerStateInformation<int> state_discrepancy;
    unsigned long long next_insertion_id;

    bool has_incumbent;
    int incumbent_cost;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    bool evaluate_successor(
        const SearchNode &parent_node, const OperatorProxy &op,
        const State &state, SearchNode &node, int succ_g, int &rank_h_out);
    void push_open(const State &state, int total_discrepancy, int g);
    void update_incumbent(const State &goal_state);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    GraphDiscrepancySearch(
        const std::shared_ptr<Evaluator> &eval,
        const std::shared_ptr<Evaluator> &prune_eval, bool reopen_closed,
        bool anytime, const std::string &discrepancy_mode,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~GraphDiscrepancySearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
