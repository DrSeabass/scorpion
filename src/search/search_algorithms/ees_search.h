#ifndef SEARCH_ALGORITHMS_EES_SEARCH_H
#define SEARCH_ALGORITHMS_EES_SEARCH_H

#include "../per_state_information.h"
#include "../search_algorithm.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace ees_search {

/*
  Explicit Estimation Search (EES; Thayer & Ruml, IJCAI 2011), "Bounded
  Suboptimal Search: A Direct Approach Using Inadmissible Estimates". EES is a
  bounded-suboptimal best-first search that separates the roles of the three
  estimators it consults:
    - h  : an admissible heuristic. f = g + h is a lower bound; it orders the
           `cleanup` list and its minimum f_min drives the w-bound proof.
    - hhat (h^): a (potentially inadmissible) estimate of the cost-to-go. f^ =
           g + h^ is EES's guess at solution cost; it orders the `open` list.
    - dhat (d^): a (potentially inadmissible) estimate of the search
           distance-to-go (actions remaining). It orders the `focal` list.

  At each step selectNode picks one of three candidates:
    - best_f^  = argmin_{n in open}      f^(n)          (open.begin())
    - best_d^  = argmin_{n in focal}     d^(n)          (focal.begin()), where
                 focal = { n in open : f^(n) <= w * f^(best_f^) }
    - best_f   = argmin_{n in cleanup}   f(n) = f_min   (cleanup.begin())
  via:
    1. if   f^(best_d^) <= w * f(best_f)  ->  best_d^
    2. elif f^(best_f^) <= w * f(best_f)  ->  best_f^
    3. else                               ->  best_f

  Theorem 1 of the paper shows every node EES selects satisfies f(n) <=
  w * g(opt). A goal has h = 0, so f = g; hence the first goal selectNode
  chooses costs at most w * g(opt) (this holds regardless of h^ >= h, since a
  goal is only picked when its g is bounded by w * f_min <= w * g(opt)). EES is
  therefore a plain best-first search that returns the first goal it *selects*
  -- no anytime / incumbent machinery is needed. dhat defaults to hhat.

  With `debias` on, h^ and d^ are not read straight from the hhat/dhat
  evaluators but *corrected* for the systematic single-step error of those base
  estimates, following Thayer, Dionne & Ruml (2011), "Learning inadmissible
  heuristics during search". Here we use the path-based variant: each node
  inherits the running mean single-step error along its root->node path:
    eps_d = mean over path edges of  (1 + d_base(child)) - d_base(parent)
    eps_h = mean over path edges of  (cost(op) + h_base(child)) - h_base(parent)
  and corrects the base estimates by
    d^(n) = d_base(n) / (1 - eps_d)     [eps_d extra steps per remaining step]
    h^(n) = h_base(n) + d^(n) * eps_h   [eps_h extra cost per remaining step]
  turning an admissible/base estimate into a sharper (inadmissible) one. Off by
  default, so h^ = hhat and d^ = dhat verbatim.

  Scorpion adaptations mirror the sibling ports (bsor_search): get_g() stands in
  for the absent get_real_g() (equal under NORMAL cost type) and trace_path
  takes (task_proxy, successor_generator, state).
*/
class EESSearch : public SearchAlgorithm {
    const double w;
    const bool debias;

    // h: admissible, orders `cleanup` and proves the bound (f = g + h).
    // hhat: base (potentially inadmissible) cost estimate, orders `open`.
    // dhat: base distance-to-go estimate, orders `focal` (defaults to hhat).
    std::shared_ptr<Evaluator> h_evaluator;
    std::shared_ptr<Evaluator> hhat_evaluator;
    std::shared_ptr<Evaluator> dhat_evaluator;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    // A frontier entry is (ordering value, unique per-state sequence number).
    // In `open` the value is f^ = g + hhat; in `cleanup` it is f = g + h; in
    // `focal` it is dhat. The seq breaks ties and keys the entry to a state.
    using Entry = std::pair<int, int>;
    std::set<Entry> open;
    std::set<Entry> cleanup;
    std::set<Entry> focal;
    // focal == { entries of `open` in [open.begin(), focal_boundary) }. Kept in
    // sync so focal.begin() is best_d^; see sync_focal / open_insert /
    // open_erase.
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
    bool is_before_boundary(const Entry &e) const;
    void open_insert(const Entry &e_open, const Entry &e_focal);
    void open_erase(const Entry &e_open, const Entry &e_focal);
    void frontier_insert(const State &state);
    void frontier_erase(const State &state);
    void sync_focal();
    int select_node();
    void expand(const State &state);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    EESSearch(
        const std::shared_ptr<Evaluator> &h,
        const std::shared_ptr<Evaluator> &hhat,
        const std::shared_ptr<Evaluator> &dhat, double w, bool debias,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~EESSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
