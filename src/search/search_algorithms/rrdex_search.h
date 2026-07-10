#ifndef SEARCH_ALGORITHMS_RRDEX_SEARCH_H
#define SEARCH_ALGORITHMS_RRDEX_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace rrdex_search {

/*
  Round-Robin on Dynamic eXpected Effort Search (RR-DXES / "RR-Dex"; Fickert, Gu
  & Ruml, "New Results in Bounded Suboptimal Search", AAAI 2022). This is the
  reference fast-downward-xes `dxes` search run with alternation_mode=BOTH and
  its default variance settings.

  RR-DXES is structurally identical to RR-D (see rrd_search): a bounded-
  suboptimal best-first search that rotates through three queues in lockstep
  with the expansion counter,
    - expanded % 3 == 0  ->  argmin_{focal_ew}   expected effort
    - expanded % 3 == 1  ->  argmin_{focal_fhat} f^ = g + hhat
    - expanded % 3 == 2  ->  argmin_{cleanup}    f  = g + h  = f_min
  where both focal queues hold exactly the nodes provably within the bound by
  the admissible f, focal = { n : f(n) <= w * f_min }. The one difference from
  RR-D is the ordering of the first queue: instead of the raw distance-to-go d,
  RR-DXES orders it by *dynamic expected effort*
    ew(n) = dhat(n) / P(n leads to a within-bound solution),
  ranking cheap-to-reach nodes that are also likely to satisfy the w-bound.

  P(n) is DXES's belief computation (the paper's "Nancy assumptions", default
  variance model). Two Gaussian beliefs are combined:
    - solution cost ~ N(mean = f^(n) = g + hhat, sd = |f^(n) - f(n)| / 2)
    - cost bound    ~ N(mean = w * fhat_min,     sd = sqrt(Var(fhat_min)))
  where fhat_min is the minimum f^ over the in-bound frontier and Var(fhat_min)
  is Welford's running variance of the fhat_min sequence sampled once per
  expansion. P(n) is the probability that (cost bound - solution cost) >= 0,
  i.e. 1 - Phi(-mean / sd) of the difference distribution. Goals have dhat = 0,
  so ew = 0 and they sort first in focal_ew; the first goal RR-DXES *selects*
  from any queue has f = g <= w * f_min <= w * g(opt), hence within the bound.

  This port reproduces the reference's default configuration: expected-work keys
  are frozen at focal insertion (expected_work_error_margin < 0, no stale-key
  re-evaluation), the solution-cost spread uses |f^ - f| / 2
  (use_online_variance=false), and the cost-bound spread uses the f_min-variance
  model (F_MIN_VARIANCE). The non-default variance models and the online-
  variance heuristic-error estimator are not ported.

  Like the rrd_search / ees_search siblings it consults three estimators (h
  admissible; hhat inadmissible cost; dhat distance-to-go, default hhat) with an
  optional path-based `debias` correction, and uses get_g() for the absent
  get_real_g() (equal under NORMAL cost type). Unlike the reference -- which
  derives a single f^ by debiasing one admissible heuristic -- this port takes
  hhat explicitly, matching rrd_search so the two can be compared under
  identical evaluators. The Nancy belief math is identical given f, f^, dhat.
*/
class RRDEXSearch : public SearchAlgorithm {
    const double w;
    const bool debias;

    // h: admissible, orders `cleanup` and proves the bound (f = g + h).
    // hhat: base (potentially inadmissible) cost estimate; f^ = g + hhat orders
    //       `focal_fhat` and feeds the expected-effort belief.
    // dhat: base distance-to-go estimate; the numerator of expected effort
    //       (defaults to hhat).
    std::shared_ptr<Evaluator> h_evaluator;
    std::shared_ptr<Evaluator> hhat_evaluator;
    std::shared_ptr<Evaluator> dhat_evaluator;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    using Entry = std::pair<int, int>;
    using EWEntry = std::pair<double, int>;
    // All open nodes, ordered by admissible f. cleanup.begin() is best_f/f_min.
    std::set<Entry> cleanup;
    // focal_ew and focal_fhat share the membership { n in cleanup : f <= w*f_min }
    // but are ordered by expected effort and by f^ respectively. focal_ew.begin()
    // is the min-expected-effort node; focal_fhat.begin() is best_f^ and its key
    // is the current fhat_min. Kept in sync via focal_boundary.
    std::set<EWEntry> focal_ew;
    std::set<Entry> focal_fhat;
    // First `cleanup` entry NOT admitted into the focal queues (f > w * f_min),
    // or cleanup.end() if all are admitted.
    std::set<Entry>::iterator focal_boundary;

    // fhat_min and its Welford running variance, sampled once per expansion.
    // These feed the cost-bound belief; they are updated before a node's
    // successors are inserted, so expected-work keys are computed against the
    // fhat_min known as of the current expansion (matching the reference).
    double fhatmin_value;
    double fhatmin_mean;
    double fhatmin_m2;
    int fhatmin_count;

    struct SeqInfo {
        int h;
        int hhat_base;
        int dhat_base;
        int hhat;
        int dhat;
        int errh_sum;
        int errd_sum;
        int path_len;
        // Expected-effort key this node currently holds in focal_ew (frozen at
        // the time it was admitted), so it can be erased with the same key.
        double ew;
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
    int node_g(int seq);
    double within_bound_probability(int f, int fhat) const;
    double expected_effort(int seq);
    void sample_fhatmin();
    Entry cleanup_entry(int seq);
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
    RRDEXSearch(
        const std::shared_ptr<Evaluator> &h,
        const std::shared_ptr<Evaluator> &hhat,
        const std::shared_ptr<Evaluator> &dhat, double w, bool debias,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~RRDEXSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
