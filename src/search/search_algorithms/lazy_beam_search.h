#ifndef SEARCH_ALGORITHMS_LAZY_BEAM_SEARCH_H
#define SEARCH_ALGORITHMS_LAZY_BEAM_SEARCH_H

#include "../search_algorithm.h"

#include <memory>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace lazy_beam_search {

/*
  Lazy beam search: same layer-by-layer beam expansion as eager beam search,
  but named separately to allow independent option tuning.
*/
class LazyBeamSearch : public SearchAlgorithm {
    const int beam_width;
    std::shared_ptr<Evaluator> eval;
    std::shared_ptr<PruningMethod> pruning_method;

    std::vector<Evaluator *> path_dependent_evaluators;
    std::vector<StateID> beam;

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    LazyBeamSearch(
        const std::shared_ptr<Evaluator> &eval, int beam_width,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~LazyBeamSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
