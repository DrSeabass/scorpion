#ifndef SEARCH_ALGORITHMS_ADAPTIVE_RECTANGLE_SEARCH_H
#define SEARCH_ALGORITHMS_ADAPTIVE_RECTANGLE_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

class Evaluator;
class PruningMethod;

namespace adaptive_rectangle_search {

/*
  Adaptive Rectangle Search: a fork of rectangle_search (Lemons, Ruml, Holte,
  Sturtevant, AAAI 2024) whose single parameter -- the aspect ratio a -- is set
  dynamically instead of fixed. There is no beam width and no static aspect;
  the rectangle "rotates" as a changes.

  The traversal is identical to the faithful base: the frontier is bucketed by
  search depth `de` into `rect`, ordered within a depth by `eval`, and the
  rectangle grows with the `iteration` counter (each depth level admits
  iteration * delta_across expansions; the rectangle reaches depth
  iteration * delta_down). The only difference is that (delta_down,
  delta_across) are recomputed from a live aspect `a` -- (a, 1) if a >= 1 else
  (1, 1/a) -- rather than a constant.

  a is driven by a parameter-free best-first-chain ratchet, a two-way ratchet in
  the spirit of ratchet_triangle (which doubles/halves its slope each step on an
  informed-vs-uninformed count). The signal here is structural: within a sweep,
  the first (best-eval) node expanded at a depth level -- the head of that
  level's beam -- votes to deepen iff the front (best) of the next depth's
  bucket is one of the successors it just produced. That is, the greedy
  best-first chain stays intact from one depth to the next: the beam's best node
  hands the next beam its best node. A coherent chain means a deep-narrow
  rectangle is tracking a real gradient toward the goal; a broken chain means
  the best path is scattering across the frontier, so a wider sweep is better.
  Over each completed rectangle sweep (an `iteration`) these per-level votes are
  tallied and the ratchet fires at the boundary:
    chain-intact votes strictly dominate -> a *= 2  (rotate deeper)
    chain-broken votes strictly dominate -> a /= 2  (rotate wider)
    tie / no data                        -> hold
  a is always a power of two, clamped to [1/1024, 1024] as a safety rail (not a
  tuning knob). Completeness/optimality do not depend on a: iteration grows
  without bound and every depth level keeps being served, so the dynamic aspect
  changes only the expansion order.

  adaptive_triangle / ratchet_triangle are the width-1 relatives of this search.
*/
class AdaptiveRectangleSearch : public SearchAlgorithm {
    const bool reopen_closed_nodes;
    const bool anytime_search;

    std::shared_ptr<Evaluator> eval;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    // Live aspect ratio and its per-iteration depth (down) / width (across)
    // split: (a, 1) if a >= 1 else (1, 1/a). Recomputed whenever the ratchet
    // changes `aspect`.
    double aspect;
    double delta_down;
    double delta_across;

    // Best-first-chain ratchet state. Per rectangle sweep, the first (best)
    // node expanded at each depth level casts one vote: chain-intact if the
    // front of the next depth's bucket is one of the successors it produced,
    // chain-broken otherwise. The tallies drive the aspect ratchet at the sweep
    // (iteration) boundary and reset there. `spine_level` is the deepest level
    // that has already voted this sweep, so each level votes once (levels are
    // served in increasing order within a sweep); reset to -1 at the boundary.
    int chain_intact_votes;
    int chain_broken_votes;
    int spine_level;

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
    void recompute_deltas();
    void apply_aspect_ratchet();
    bool advance_rectangle();
    bool expand(const State &state, bool first_in_beam);
    bool try_improve_incumbent(
        const State &parent_state, OperatorID op_id, int succ_g);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    AdaptiveRectangleSearch(
        const std::shared_ptr<Evaluator> &eval, bool reopen_closed,
        bool anytime, const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~AdaptiveRectangleSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
