#include "lazy_beam_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_lazy_beam {
class LazyBeamSearchFeature
    : public plugins::TypedFeature<
          SearchAlgorithm, lazy_beam_search::LazyBeamSearch> {
public:
    LazyBeamSearchFeature() : TypedFeature("lazy_beam") {
        document_title("Lazy beam search");
        document_synopsis(
            "Beam search with deferred evaluation. Uses the same layer-by-layer "
            "expansion as eager beam search but defers heuristic evaluation.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<int>(
            "beam_width", "maximum number of states kept in each beam layer",
            "100", plugins::Bounds("1", "infinity"));
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "lazy_beam");
    }

    virtual shared_ptr<lazy_beam_search::LazyBeamSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            lazy_beam_search::LazyBeamSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<int>("beam_width"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyBeamSearchFeature> _plugin;
}
