#include "bead_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_bead {
class BeadSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, bead_search::BeadSearch> {
public:
    BeadSearchFeature() : TypedFeature("bead") {
        document_title("Bead search");
        document_synopsis(
            "Bead (Lemons et al. 2022): an unboundedly-suboptimal beam search "
            "that ranks the beam by a distance-to-go estimate and drops "
            "duplicate states rather than reopening them.");

        add_option<shared_ptr<Evaluator>>(
            "eval", "distance-to-go ranking evaluator (d^-proxy, e.g. ff())");
        add_option<int>(
            "beam_width", "maximum number of states kept in each beam layer",
            "100", plugins::Bounds("1", "infinity"));
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "bead");
    }

    virtual shared_ptr<bead_search::BeadSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<bead_search::BeadSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<int>("beam_width"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<BeadSearchFeature> _plugin;
}
