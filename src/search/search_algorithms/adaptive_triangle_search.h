#ifndef SEARCH_ALGORITHMS_ADAPTIVE_TRIANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_ADAPTIVE_TRIANGLE_SEARCH_H

#include "../search_algorithm.h"

#include <deque>
#include <fstream>
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
    // Budget decrement applied per uninformed (h non-improving) transition.
    // Default 1 matches the original symmetric +1/-1 budget rule. Setting to
    // 0 lets the cascade keep running through non-improving transitions as
    // long as no frontier extension is needed -- so a single bad transition
    // mid-step no longer halts forward progress.
    const int non_progress_penalty;
    // When true, append one CSV row per expansion (expansions,budget) to the
    // hardcoded file adaptive_triangle_budget.csv, tracing the per-expansion
    // depth budget. Off => no file is opened and the hot loop pays only one
    // predicted branch.
    const bool log_budget;

    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<Evaluator> pruning_heuristic;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::deque<OpenList> open_lists;
    int max_active_layer = -1;
    // Absolute depth of open_lists[0]: incremented whenever an empty front
    // layer is popped, so reported depths stay absolute despite front-draining.
    int depth_offset = 0;
    // Number of new frontier layers the previous step instantiated (its
    // realized dive depth past the frontier -- the emergent analog of
    // ratchet's persistent slope). Drives the LAYERS_ADDED floor proxy.
    int prev_layers_added = 0;
    // Improving / non-improving h-transition counts for the INFORMEDNESS floor
    // proxy. Persistent across steps, reset on each incumbent improvement so
    // the floor reflects the current epoch's heuristic quality.
    int improving_transitions = 0;
    int nonimproving_transitions = 0;
    // Per-step heuristic-trend budget. Unspent budget carries between steps:
    // at the top of each step it is reset to max(1, remaining), so a step
    // that ended with leftover credit starts richer while a depleted step
    // still gets one unit to make forward progress.
    int budget = 1;

    // Open when log_budget is set; one row per expansion.
    std::ofstream budget_log_file;
    // Open when log_budget is set; one row index per incumbent improvement,
    // marking the extension-check log element at which a solution was found.
    std::ofstream budget_solution_file;
    // Count of rows written to budget_log_file so far (one per frontier-
    // extension check); used to place solution markers on that same axis.
    int budget_log_rows = 0;

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
    AdaptiveTriangleSearch(
        const std::shared_ptr<Evaluator> &eval,
        bool reopen_closed,
        bool anytime,
        bool lift_floor,
        FloorProxy floor_proxy,
        int non_progress_penalty,
        bool log_budget,
        const std::shared_ptr<Evaluator> &pruning_heuristic,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~AdaptiveTriangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
