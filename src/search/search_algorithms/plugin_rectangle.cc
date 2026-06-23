#include "rectangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_rectangle {
class RectangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, rectangle_search::RectangleSearch> {
public:
    RectangleSearchFeature() : TypedFeature("rectangle") {
        document_title("Rectangle search");
        document_synopsis(
            "Rectangle search - an anytime beam search variant. "
            "Implements the algorithm from: "
            "\"Rectangle Search: An Anytime Beam Search\" by Lemons et al. (AAAI 2024). "
            "Maintains multiple open lists (one per depth layer) and uses a specific "
            "expansion pattern controlled by the 'aspect' parameter.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<int>(
            "beam_width",
            "number of expansion attempts per selected open list",
            "100",
            plugins::Bounds("1", "infinity"));
        add_option<int>(
            "aspect",
            "controls the depth progression rate (depth increases by aspect each iteration)",
            "1",
            plugins::Bounds("1", "infinity"));
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent "
            "(converges to the optimal cost)",
            "false");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "rectangle");
    }

    virtual shared_ptr<rectangle_search::RectangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<rectangle_search::RectangleSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<int>("beam_width"),
            opts.get<int>("aspect"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<RectangleSearchFeature> _plugin;
}
