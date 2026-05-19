#ifndef SEARCH_ALGORITHMS_BEAM_SEARCH_H
#define SEARCH_ALGORITHMS_BEAM_SEARCH_H

#include "../search_algorithm.h"

#include <memory>
#include <string>
#include <vector>

class Evaluator;
class PruningMethod;

namespace beam_search {

class BeamSearch : public SearchAlgorithm {
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
    BeamSearch(
        const std::shared_ptr<Evaluator> &eval, int beam_width,
        const std::shared_ptr<PruningMethod> &pruning, OperatorCost cost_type,
        int bound, double max_time, const std::string &description,
        utils::Verbosity verbosity);
    virtual ~BeamSearch() = default;

    virtual void print_statistics() const override;
};

}

#endif
