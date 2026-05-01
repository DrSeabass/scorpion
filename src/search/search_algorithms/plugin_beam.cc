#include "beam_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_beam {
class BeamSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, beam_search::BeamSearch> {
public:
    BeamSearchFeature() : TypedFeature("beam") {
        document_title("Beam search");
        document_synopsis("Eager beam-search over states.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<int>(
            "beam_width",
            "maximum number of states kept in each beam layer",
            "100",
            plugins::Bounds("1", "infinity"));
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "beam");
    }

    virtual shared_ptr<beam_search::BeamSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<beam_search::BeamSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<int>("beam_width"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<BeamSearchFeature> _plugin;
}
