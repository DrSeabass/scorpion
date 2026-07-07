#ifndef SEARCH_ALGORITHMS_BSOR_SEARCH_H
#define SEARCH_ALGORITHMS_BSOR_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

class Evaluator;
class PruningMethod;

namespace bsor_search {

/*
  Bounded-Suboptimal Rectangle Search (Thomas, Wissow, Bauer, McAfee, Ruml,
  HSDIP 2026), Algorithms 4 and 5. BSOR adapts Rectangle Search (an anytime
  beam search) to the bounded-suboptimal setting: it keeps an A*-style open
  list ordered on f = g + h alongside the rectangle `rect` (nodes bucketed by
  search depth `de` and, within a depth, ordered by a distance-to-go estimate
  d). Rectangle expansions walk the rectangle deeper-and-wider per the aspect
  ratio; the search returns the incumbent once it is provably within the
  suboptimality bound (w * f_min >= g(incumbent)).

  Setting rr=true yields Round-Robin Rectangle Search (RRR), which interleaves
  a lowest-f (min_f) expansion from open before each rectangle expansion to
  raise f_min faster.

  `eval` is the cost estimate h (so f = g + eval), used for open ordering and
  the bound; `dist` is the distance-to-go estimate d used for the within-depth
  rect ordering. They may differ (e.g. eval=ff(), dist=lmcut()); dist defaults
  to eval. The bounded-suboptimality guarantee only holds for admissible eval,
  but any evaluator is accepted.
*/
class BSORSearch : public SearchAlgorithm {
    const double w;
    const double aspect;
    const bool round_robin;

    // h estimate: f = g + eval, orders `open` and drives the suboptimality
    // bound. d estimate: distance-to-go proxy that orders each `rect` depth
    // bucket. They may be the same evaluator (dist defaults to eval).
    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<Evaluator> dist;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    // Aspect ratio split into the per-iteration depth (down) and width (across)
    // allowances: (a, 1) if a >= 1 else (1, 1/a).
    double delta_down;
    double delta_across;

    // A frontier entry is (ordering value, unique per-state sequence number).
    // In `open` the value is f = g + h; in a `rect` level it is the d estimate.
    using Entry = std::pair<int, int>;
    std::set<Entry> open;
    std::vector<std::set<Entry>> rect;
    std::vector<int> ec;
    int iteration;
    int level;

    // Per-state bookkeeping. `seq` gives each generated state a stable unique
    // id so it can key the ordered frontiers; `seq_to_state` maps it back.
    PerStateInformation<int> node_seq;
    PerStateInformation<int> node_de;
    PerStateInformation<int> node_h;
    PerStateInformation<int> node_d;
    std::vector<StateID> seq_to_state;
    int next_seq;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    int assign_seq(const State &state);
    void ensure_level(int idx);
    bool has_non_empty_rect() const;
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
    BSORSearch(
        const std::shared_ptr<Evaluator> &eval,
        const std::shared_ptr<Evaluator> &dist, double w, double aspect,
        bool round_robin, const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~BSORSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
