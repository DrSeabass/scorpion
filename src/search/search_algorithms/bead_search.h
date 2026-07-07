#ifndef SEARCH_ALGORITHMS_BEAD_SEARCH_H
#define SEARCH_ALGORITHMS_BEAD_SEARCH_H

#include "../search_algorithm.h"

#include <memory>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace bead_search {

/*
  Bead (Lemons, Lopez, Holte, and Ruml 2022): an unboundedly-suboptimal beam
  search. It is the generic beam search (see beam_search) with two defining
  traits:
    - the beam is ranked by a distance-to-go estimate (d^) rather than by cost
      (f/h); in this port the ranking evaluator is whatever `eval` is passed,
      matching the triangle/rectangle ports (e.g. ff() or lmcount as a
  d^-proxy).
    - it DROPS duplicate states rather than reopening/updating them on a cheaper
      path. This keeps the search in the unboundedly-suboptimal regime;
  retaining duplicates would be needed for bounded-suboptimality (cf. BSBS).
*/
class BeadSearch : public SearchAlgorithm {
    const int beam_width;

    std::shared_ptr<Evaluator> eval;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::shared_ptr<PruningMethod> pruning_method;

    std::vector<StateID> beam;

    void start_evaluator_statistics(EvaluationContext &eval_context);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    BeadSearch(
        const std::shared_ptr<Evaluator> &eval, int beam_width,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~BeadSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
