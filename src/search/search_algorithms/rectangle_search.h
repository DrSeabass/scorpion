#ifndef SEARCH_ALGORITHMS_RECTANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_RECTANGLE_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

class Evaluator;
class PruningMethod;

namespace rectangle_search {

/*
  Rectangle Search (Lemons, Ruml, Holte, Sturtevant, AAAI 2024): an anytime
  beam search parameterized only by the aspect ratio. The frontier is bucketed
  by search depth `de` into the rectangle `rect`; within a depth the nodes are
  ordered by the evaluator `eval` (used both as the distance-to-go proxy for
  the within-depth ordering and, via g + eval, for incumbent pruning).

  A single aspect ratio a is split into per-iteration depth (down) and width
  (across) allowances: (delta_down, delta_across) = (a, 1) if a >= 1 else
  (1, 1/a). The rectangle grows with the `iteration` counter -- the width
  visited per depth level is iteration * delta_across and the depth reached is
  iteration * delta_down -- so there is no fixed beam width. Expansions walk
  the rectangle deeper-and-wider per the aspect ratio; the search keeps
  improving the incumbent until the rectangle is exhausted or time runs out
  (anytime), or stops at the first solution (anytime=false).

  This is the anytime sibling of BSORSearch (bsor_search.h): the traversal is
  the same, but Rectangle Search has no bounded-suboptimal termination and no
  round-robin, and it exposes a single evaluator.
*/
class RectangleSearch : public SearchAlgorithm {
    const double aspect;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    // If true, prune against the incumbent using f = g + h (only sound when
    // `eval` is admissible); if false, prune using g alone.
    const bool prune_with_h;

    std::shared_ptr<Evaluator> eval;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    // Aspect ratio split into the per-iteration depth (down) and width (across)
    // allowances: (a, 1) if a >= 1 else (1, 1/a).
    double delta_down;
    double delta_across;

    // A frontier entry is (eval value, unique per-state sequence number). The
    // eval value orders each `rect` depth bucket.
    using Entry = std::pair<int, int>;
    std::vector<std::set<Entry>> rect;
    std::vector<int> ec;
    int iteration;
    int level;

    // Per-state bookkeeping. `seq` gives each generated state a stable unique
    // id so it can key the ordered frontiers; `seq_to_state` maps it back.
    PerStateInformation<int> node_seq;
    PerStateInformation<int> node_de;
    PerStateInformation<int> node_h;
    std::vector<StateID> seq_to_state;
    int next_seq;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    int assign_seq(const State &state);
    void ensure_level(int idx);
    bool has_non_empty_rect() const;
    void frontier_insert(const State &state);
    void frontier_erase(const State &state);
    bool advance_rectangle();
    bool expand(const State &state);
    bool try_improve_incumbent(
        const State &parent_state, OperatorID op_id, int succ_g);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    RectangleSearch(
        const std::shared_ptr<Evaluator> &eval, double aspect,
        bool reopen_closed, bool anytime, bool prune_with_h,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~RectangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
