#ifndef SEARCH_ALGORITHMS_RRD_SEARCH_H
#define SEARCH_ALGORITHMS_RRD_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace rrd_search {

/*
  Round-Robin on distance (RR-D; Fickert, Gu & Ruml, "New Results in Bounded
  Suboptimal Search", AAAI 2022). RR-D is a bounded-suboptimal best-first search
  that, like EES (see ees_search), consults three estimators:
    - h  : an admissible heuristic. f = g + h is a lower bound; it orders the
           `cleanup` list, and its minimum f_min drives the w-bound proof.
    - hhat (h^): a (potentially inadmissible) estimate of solution cost. f^ =
           g + h^ orders the `focal_fhat` list.
    - dhat (d^): a (potentially inadmissible) estimate of the search
           distance-to-go (actions remaining). It orders the `focal_d` list.
           Defaults to hhat.

  Where EES picks among the three candidates by a priority cascade, RR-D simply
  *rotates* through the three queues in lockstep with the expansion counter:
    - expanded % 3 == 0  ->  argmin_{focal_d}    d^   (the "D" in RR-D)
    - expanded % 3 == 1  ->  argmin_{focal_fhat} f^
    - expanded % 3 == 2  ->  argmin_{cleanup}    f = f_min

  Both focal lists hold exactly the nodes provably within the bound by the
  admissible f, i.e. focal = { n : f(n) = g(n) + h(n) <= w * f_min }. (This is
  the reference fast-downward-xes implementation's rule, and differs from EES,
  whose focal is bounded by f^ rather than by the admissible f.) Because every
  focal node -- and the cleanup minimum -- satisfies f <= w * f_min <= w *
  g(opt), any goal RR-D *selects* from any of the three queues (a goal has
  h = 0, so f = g) costs at most w * g(opt). RR-D is therefore a plain
  best-first search that returns the first goal it selects: no anytime /
  incumbent machinery is needed.

  With `debias` on, h^ and d^ are corrected for the systematic single-step error
  of the base estimates along each node's root->node path (Thayer, Dionne &
  Ruml 2011), turning them into sharper inadmissible estimates; off by default,
  so h^ = hhat and d^ = dhat verbatim. The correction is identical to the EES
  port's; see ees_search for the formulae.

  Scorpion adaptations mirror the sibling ports (ees_search, bsor_search):
  get_g() stands in for the absent get_real_g() (equal under NORMAL cost type)
  and trace_path takes (task_proxy, successor_generator, state). The reference
  fork registers this as `alt_d` ("alternating" on distance).
*/
class RRDSearch : public SearchAlgorithm {
    const double w;
    const bool debias;

    // h: admissible, orders `cleanup` and proves the bound (f = g + h).
    // hhat: base (potentially inadmissible) cost estimate, orders `focal_fhat`.
    // dhat: base distance-to-go estimate, orders `focal_d` (defaults to hhat).
    std::shared_ptr<Evaluator> h_evaluator;
    std::shared_ptr<Evaluator> hhat_evaluator;
    std::shared_ptr<Evaluator> dhat_evaluator;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    // A frontier entry is (ordering value, unique per-state sequence number).
    // In `cleanup` the value is f = g + h; in `focal_fhat` it is f^ = g + hhat;
    // in `focal_d` it is d^. The seq breaks ties and keys the entry to a state.
    using Entry = std::pair<int, int>;
    // All open nodes, ordered by admissible f. cleanup.begin() is best_f/f_min.
    std::set<Entry> cleanup;
    // focal_d and focal_fhat hold the same membership -- the prefix of `cleanup`
    // with f <= w * f_min -- but ordered by d^ and f^ respectively, so
    // focal_d.begin() is best_d^ and focal_fhat.begin() is best_f^. Kept in sync
    // via focal_boundary; see sync_focal / cleanup_insert / cleanup_erase.
    std::set<Entry> focal_d;
    std::set<Entry> focal_fhat;
    // First `cleanup` entry NOT admitted into the focal lists (f > w * f_min),
    // or cleanup.end() if all are admitted.
    std::set<Entry>::iterator focal_boundary;

    // Per-state bookkeeping keyed by a stable per-state seq number. `h` and the
    // *_base estimates are fixed per state; `hhat`/`dhat` are the effective
    // (possibly debiased) estimates used in the frontier keys, and the err/path
    // fields accumulate the path-based single-step error when debias is on.
    struct SeqInfo {
        int h;
        int hhat_base;
        int dhat_base;
        int hhat;
        int dhat;
        int errh_sum;
        int errd_sum;
        int path_len;
    };
    PerStateInformation<int> state_seq;
    std::vector<StateID> seq_to_state;
    std::vector<SeqInfo> info;
    int next_seq;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    int register_node(
        const State &state, int h, int hhat_base, int dhat_base, int parent_seq,
        int step_cost);
    void reparent_node(int seq, int parent_seq, int step_cost);
    void apply_correction(int seq);
    Entry cleanup_entry(int seq);
    Entry focal_d_entry(int seq);
    Entry focal_fhat_entry(int seq);
    bool is_before_boundary(const Entry &e) const;
    void frontier_insert(const State &state);
    void frontier_erase(const State &state);
    void sync_focal();
    int select_node();
    void expand(const State &state);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    RRDSearch(
        const std::shared_ptr<Evaluator> &h,
        const std::shared_ptr<Evaluator> &hhat,
        const std::shared_ptr<Evaluator> &dhat, double w, bool debias,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~RRDSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
