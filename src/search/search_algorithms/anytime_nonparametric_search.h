#ifndef SEARCH_ALGORITHMS_ANYTIME_NONPARAMETRIC_SEARCH_H
#define SEARCH_ALGORITHMS_ANYTIME_NONPARAMETRIC_SEARCH_H

#include "../search_algorithm.h"

#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace anytime_nonparametric_search {

// Anytime Nonparametric A* (ANA*), van den Berg, Shah, Huang & Goldberg,
// AAAI 2011. Eager best-first search that expands the open node with the
// MAXIMAL potential
//
//     e(s) = (G - g(s)) / h(s),
//
// where G is the cost of the current incumbent (initially infinity). The
// ordering needs no weight parameter: with G = infinity the first-solution
// search is maximally greedy (h-only, like GBFS); as G tightens after each
// incumbent, the search automatically re-greedifies toward improving it.
// Nodes with g(s) + h(s) >= G (equivalently e(s) <= 1) cannot improve the
// incumbent and are pruned. Because e(s) depends on G, the open list is
// re-keyed (rebuilt) whenever a new incumbent lowers G.
class AnytimeNonparametricSearch : public SearchAlgorithm {
    struct OpenEntry {
        StateID id;
        int g;
        int h;
    };

    // Orders entries by potential e(s) = (G - g)/h, highest first; ties broken
    // toward lower g. h == 0 yields e = +infinity (top priority). Reads the
    // current incumbent bound G through a pointer so a single comparator type
    // serves every rebuild; the heap is only valid while G is constant, so the
    // open list is rebuilt on each incumbent improvement (see reorder_open()).
    struct OpenEntryCompare {
        const int *bound;

        bool operator()(const OpenEntry &lhs, const OpenEntry &rhs) const {
            // priority_queue pops the greatest element, so return true when
            // lhs has LOWER priority (should be expanded after rhs).
            const long long g = *bound;
            const bool lhs_inf = (lhs.h == 0);
            const bool rhs_inf = (rhs.h == 0);
            if (lhs_inf || rhs_inf) {
                if (lhs_inf && rhs_inf)
                    return lhs.g > rhs.g;
                // Exactly one has e = +infinity; the finite one ranks lower.
                return !lhs_inf;
            }
            // Both finite with h > 0: e(lhs) < e(rhs)
            // <=> (G - lhs.g) / lhs.h < (G - rhs.g) / rhs.h
            // <=> (G - lhs.g) * rhs.h < (G - rhs.g) * lhs.h  (h values > 0).
            const long long lhs_key = (g - lhs.g) * static_cast<long long>(rhs.h);
            const long long rhs_key = (g - rhs.g) * static_cast<long long>(lhs.h);
            if (lhs_key != rhs_key)
                return lhs_key < rhs_key;
            return lhs.g > rhs.g;
        }
    };

    using OpenList =
        std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryCompare>;

    const bool reopen_closed_nodes;
    const bool anytime_search;
    // If true, prune against the incumbent using f = g + h (only sound when
    // `eval` is admissible); if false, prune using g alone.
    const bool prune_with_h;

    std::shared_ptr<Evaluator> eval;
    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    OpenList open_list;
    // Tightest suboptimality bound proven so far (the e-value of the most
    // recently expanded node once an incumbent exists); for reporting only.
    double suboptimality_bound = std::numeric_limits<double>::infinity();

    void start_evaluator_statistics(EvaluationContext &eval_context);
    void update_incumbent(const State &goal_state, bool &improved);
    void reorder_open();
    bool evaluate_and_prepare_node(
        const State &state, SearchNode &node, int g, int &h_out,
        bool is_new_evaluation);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    AnytimeNonparametricSearch(
        const std::shared_ptr<Evaluator> &eval,
        bool reopen_closed,
        bool anytime,
        bool prune_with_h,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~AnytimeNonparametricSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
