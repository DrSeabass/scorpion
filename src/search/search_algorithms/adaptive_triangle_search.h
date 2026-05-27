#ifndef SEARCH_ALGORITHMS_ADAPTIVE_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_ADAPTIVE_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace adaptive_triangle_search {

// Direction B floor proxy: how the lifted cascade start-depth is derived.
enum class FloorProxy {
    // Previous step's realized dive depth (number of new frontier layers it
    // instantiated). Conservative; floor = min(prev_layers_added - 1, ...).
    LAYERS_ADDED,
    // Position the floor between root and frontier by the fraction of recent
    // transitions that improved h: floor = max_active * I / (I + N), counts
    // reset on each incumbent improvement.
    INFORMEDNESS
};

class AdaptiveTriangleSearch : public SearchAlgorithm {
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

    const bool reopen_closed_nodes;
    const bool anytime_search;
    // Direction B (relaxed cascade start-depth): when true, begin each step's
    // cascade above the root instead of at index 0, by the floor_proxy rule.
    // Off => start index stays 0 => bit-identical to vanilla adaptive.
    const bool lift_floor;
    const FloorProxy floor_proxy;

    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<Evaluator> pruning_heuristic;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::deque<OpenList> open_lists;
    int max_active_layer = -1;
    // Number of new frontier layers the previous step instantiated (its
    // realized dive depth past the frontier -- the emergent analog of
    // ratchet's persistent slope). Drives the LAYERS_ADDED floor proxy.
    int prev_layers_added = 0;
    // Improving / non-improving h-transition counts for the INFORMEDNESS floor
    // proxy. Persistent across steps, reset on each incumbent improvement so
    // the floor reflects the current epoch's heuristic quality.
    int improving_transitions = 0;
    int nonimproving_transitions = 0;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void extend_open_lists(int num_lists);
    void recompute_max_active_layer();
    void update_incumbent(const State &goal_state);
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g, int &h_out,
        bool is_new_evaluation);
    void insert_into_open_list(int list_index, const OpenEntry &entry);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    AdaptiveTriangleSearch(
        const std::shared_ptr<Evaluator> &eval,
        bool reopen_closed,
        bool anytime,
        bool lift_floor,
        FloorProxy floor_proxy,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~AdaptiveTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
