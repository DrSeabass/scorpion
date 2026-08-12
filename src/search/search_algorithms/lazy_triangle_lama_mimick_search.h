#ifndef SEARCH_ALGORITHMS_LAZY_TRIANGLE_LAMA_MIMICK_SEARCH_H
#define SEARCH_ALGORITHMS_LAZY_TRIANGLE_LAMA_MIMICK_SEARCH_H

#include "../evaluator.h"
#include "../open_list.h"
#include "../open_list_factory.h"
#include "../search_algorithm.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

class PruningMethod;

namespace lazy_triangle_lama_mimick_search {

/*
  Lazy triangle-LAMA-mimick search (see triangle_lama_mimick_search.h and
  icaps-27-lazy-eval-design.md / icaps-27-lazy-eval-implementation-prompt.md):
  the lazy sibling of triangle_lama_mimick_search, built on this branch's
  established lazy mechanics (see lazy_boosted_triangle_search.h) rather
  than the eager {id,h,g}-priority-queue base: successors are ranked by
  their parent's already-known h (reused for free via EvaluationContext's
  copy constructor) and are only evaluated for real -- with all N guidance
  heuristics -- when popped for expansion. Open-list entries are edges
  (predecessor_id, operator_id), so the priority bookkeeping below always
  reads a freshly-computed real h/reports real progress at the moment a
  state is actually expanded, never a value read back off an entry.

  Selection mechanism (unchanged from the eager sibling, see its header for
  the full rationale): every list -- each guidance heuristic's list, each
  helpful/preferred-only list, and (unlike the eager sibling) no pruner
  list, since this lazy family never grows a guide_by_pruning option (see
  lazy_boosted_triangle_search.h) -- carries one persistent, global (not
  per-layer) priority counter, all starting at 0
  (alternation_open_list::AlternationOpenList's `priorities`). The served
  list for a whole cascade dive is decided once per step(), before the
  cascade loop: the lowest-counter list wins, ties keep the lowest index
  (AlternationOpenList::remove_min's rule). Every expansion the served list
  actually serves costs it one point.

  The boost (LAMA's actual mechanism): whenever a state's real evaluation
  -- now happening at pop time, not generation time -- reports a new
  global-best value for any progress-tracked evaluator, every
  helpful/preferred-only list's counter drops by `boost_amount`, exactly
  like AlternationOpenList::boost_preferred(). This never fires for the
  initial state's own evaluation, matching real LAMA's LazySearch::step()
  (progress is only ever rewarded on states popped for expansion, and the
  initial state's root-opening path in LazySearch is exactly this: no
  reward) and matching the eager sibling's own "no reward from the initial
  evaluation" rule -- enforced here by an explicit is_root guard in
  process_candidate, since the eager sibling gets this for free by never
  routing the initial state through evaluate_and_prepare_node at all, and
  this lazy sibling routes every state (root included) through the one
  shared process_candidate.

  Point of interest for the design record: the eager sibling's own header
  comment flags itself as "a deliberate translation, not a literal copy"
  of LAMA precisely because real LAMA checks/rewards progress at pop time
  (its heuristics are lazy) while the eager port checks at generation time
  (its heuristics are eager). This lazy port closes exactly that gap --
  process_candidate's progress check now genuinely happens at pop/expansion
  time, the same moment real LAMA's own LazySearch::step() checks it.

  preferred_evals=[] (default) leaves no helpful lists to ever boost, so
  this reduces exactly to round-robin-by-priority over the guidance lists
  -- and at num_lists == 1, to lazy_triangle(eval=evals[0], slope=slope)
  (identical expansion counts and plan), matching every other file in this
  lazy family's own reduction target.
*/
class LazyTriangleLamaMimickSearch : public SearchAlgorithm {
    enum class ExpansionOutcome {
        SKIPPED,
        EXPANDED,
        SOLVED
    };

    struct Layer {
        std::vector<std::unique_ptr<EdgeOpenList>> lists;
        explicit Layer(std::vector<std::unique_ptr<EdgeOpenList>> &&lists_)
            : lists(std::move(lists_)) {}
    };

    const int slope;
    const bool reopen_closed_nodes;
    const bool anytime_search;
    // LAMA's boost_preferred magnitude (see alternation_open_list.cc); 0
    // makes boosting inert without removing the helpful lists themselves.
    const int boost_amount;

    std::vector<std::shared_ptr<Evaluator>> evals;
    const int num_lists;
    std::vector<std::shared_ptr<Evaluator>> preferred_evals;
    const int num_preferred;
    std::vector<int> preferred_source_index;
    // Lists per layer: num_lists guidance lists, then num_preferred helpful
    // lists. No pruner list -- see lazy_boosted_triangle_search.h.
    const int total_lists;
    std::vector<std::shared_ptr<OpenListFactory>> open_list_factories;
    std::shared_ptr<PruningMethod> pruning_method;

    std::vector<Evaluator *> path_dependent_evaluators;

    std::deque<Layer> layers;
    // AlternationOpenList-style priority counters, one per list, global
    // (not per depth layer) and persistent across the whole search.
    std::vector<int> priorities;
    bool root_pending;
    int depth_offset = 0;

    void start_evaluator_statistics(EvaluationContext &eval_context);
    bool has_non_empty_lists() const;
    bool layer_empty(const Layer &layer) const;
    Layer create_layer() const;
    void extend_layers(int num_layers);
    void trim_empty_layers();
    void update_incumbent(const State &goal_state);
    // AlternationOpenList::remove_min's selection rule: lowest priority
    // counter wins, ties keep the lowest index. A once-per-dive policy
    // decision, not scoped to any specific layer -- see
    // select_available_served for per-layer emptiness.
    int select_served() const;
    // The once-per-dive pick above can be empty at a specific layer once
    // helpful lists exist. Walks the index order (wrapping) from `intended`
    // until a non-empty list at this layer is found, without re-litigating
    // the once-per-dive choice itself.
    int select_available_served(const Layer &layer, int intended) const;
    // AlternationOpenList::boost_preferred(): drop every helpful list's
    // priority by boost_amount. No-op when num_preferred == 0.
    void boost_preferred_lists();

    ExpansionOutcome process_candidate(
        const State &state,
        StateID predecessor_id,
        OperatorID operator_id,
        int g,
        int real_g,
        int source_layer_index,
        bool is_root);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    LazyTriangleLamaMimickSearch(
        const std::vector<std::shared_ptr<Evaluator>> &evals,
        int slope,
        bool reopen_closed,
        bool anytime,
        int boost_amount,
        const std::vector<std::shared_ptr<Evaluator>> &preferred_evals,
        const std::shared_ptr<PruningMethod> &pruning,
        OperatorCost cost_type, int bound, double max_time,
        const std::string &description, utils::Verbosity verbosity);
    virtual ~LazyTriangleLamaMimickSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
