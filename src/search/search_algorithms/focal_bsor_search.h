#ifndef SEARCH_ALGORITHMS_FOCAL_BSOR_SEARCH_H
#define SEARCH_ALGORITHMS_FOCAL_BSOR_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

class Evaluator;
class PruningMethod;

namespace focal_bsor_search {

/*
  Focal Bounded-Suboptimal Rectangle Search: a focal-list variant of BSOR
  (bsor_search.h). Plain BSOR keeps a single min-d bucket per search depth, so
  the beam expands the lowest-d node at a depth even when its f is far outside
  the suboptimality bound. This variant instead splits each depth level into a
  focal sub-queue -- nodes provably within the bound (f <= w * f_min_max),
  ordered by the distance-to-go d -- and a remainder sub-queue of the rest,
  ordered by f. The beam only expands from focal, so it never spends the
  aspect-limited budget on nodes that cannot participate in a within-bound
  solution (as in A*_epsilon / EES focal search).

  The bound uses the running maximum of the global-open f_min (f_min_max)
  rather than the instantaneous f_min: every open f_min lower-bounds C* for
  admissible eval, so their running max is a tighter, monotone-non-decreasing
  valid lower bound. The focal threshold therefore never dips even when an
  inconsistent h makes f_min fluctuate. The same f_min_max feeds the
  termination test (w * f_min_max >= g(incumbent)), so termination and focal
  membership share one consistent bound. Note this differs from `bsor`, which
  uses the instantaneous f_min for termination.

  Because the threshold rises during search, remainder nodes become eligible
  over time; membership is updated lazily at selection time by promoting
  now-in-bound remainder nodes into focal. The global min-f open node always
  satisfies f = f_min <= w * f_min_max, so it always lives in some level's
  focal queue -- the beam can never globally stall while open is non-empty.

  Options mirror bsor: `rr=true` interleaves a lowest-f (min_f) expansion from
  open before each rectangle expansion (orthogonal to focal). When a level's
  focal queue is empty, `focal_expand_remainder=true` expands the level's
  min-f remainder node rather than skipping the level (default: skip).

  `eval` is the cost estimate h (f = g + eval), used for open ordering, the
  focal threshold, and the bound; `dist` is the distance-to-go estimate d used
  for the within-depth focal ordering (defaults to eval). The bounded-
  suboptimality guarantee only holds for admissible eval.
*/
class FocalBSORSearch : public SearchAlgorithm {
    const double w;
    const double aspect;
    const bool round_robin;
    const bool focal_expand_remainder;

    // h estimate: f = g + eval, orders `open`, the focal threshold and the
    // suboptimality bound. d estimate: distance-to-go proxy that orders each
    // level's focal queue. They may be the same evaluator (dist defaults to
    // eval).
    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<Evaluator> dist;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    // Aspect ratio split into the per-iteration depth (down) and width (across)
    // allowances: (a, 1) if a >= 1 else (1, 1/a).
    double delta_down;
    double delta_across;

    // A frontier entry is (ordering value, unique per-state sequence number).
    // In `open` the value is f = g + h; in a `focal` level it is the d
    // estimate; in a `remainder` level it is f = g + h.
    using Entry = std::pair<int, int>;
    std::set<Entry> open;
    std::vector<std::set<Entry>> focal;
    std::vector<std::set<Entry>> remainder;
    std::vector<int> ec;
    int iteration;
    int level;

    // Running maximum of the global-open f_min: the monotone lower bound on C*
    // that defines both the focal threshold (w * f_min_max) and termination.
    int f_min_max;
    bool bound_ready;

    // Per-state bookkeeping. `seq` gives each generated state a stable unique
    // id so it can key the ordered frontiers; `seq_to_state` maps it back.
    // `node_in_focal` records which sub-queue currently holds the node so it
    // can be erased from the right one.
    PerStateInformation<int> node_seq;
    PerStateInformation<int> node_de;
    PerStateInformation<int> node_h;
    PerStateInformation<int> node_d;
    PerStateInformation<bool> node_in_focal;
    std::vector<StateID> seq_to_state;
    int next_seq;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    int assign_seq(const State &state);
    void ensure_level(int idx);
    bool has_non_empty_rect() const;
    void refresh_bound();
    double threshold() const;
    void promote_level(int idx);
    void frontier_insert(const State &state);
    void frontier_erase(const State &state);
    bool advance_rectangle();
    void expand(const State &state);
    bool try_improve_incumbent(
        const State &parent_state, OperatorID op_id, int succ_g);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    FocalBSORSearch(
        const std::shared_ptr<Evaluator> &eval,
        const std::shared_ptr<Evaluator> &dist, double w, double aspect,
        bool round_robin, bool focal_expand_remainder,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~FocalBSORSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
