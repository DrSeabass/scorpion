#include "rectangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_rectangle {
class RectangleSearchFeature
    : public plugins::TypedFeature<
          SearchAlgorithm, rectangle_search::RectangleSearch> {
public:
    RectangleSearchFeature() : TypedFeature("rectangle") {
        document_title("Rectangle search");
        document_synopsis(
            "Rectangle search - an anytime beam search parameterized only by "
            "the aspect ratio. Implements the algorithm from: "
            "\"Rectangle Search: An Anytime Beam Search\" by Lemons, Ruml, "
            "Holte, and Sturtevant (AAAI 2024). The frontier is bucketed by "
            "search depth; the rectangle grows with an iteration counter, "
            "visiting iteration * (1/aspect) nodes per depth level and reaching "
            "depth iteration * aspect (with the split (a, 1) if a >= 1 else "
            "(1, 1/a)). There is no fixed beam width.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<double>(
            "aspect",
            "aspect ratio of the rectangle. a >= 1 grows deep-and-narrow "
            "(delta_down = a, delta_across = 1); a < 1 grows wide-and-shallow "
            "(delta_down = 1, delta_across = 1/a).",
            "1", plugins::Bounds("0.0", "infinity"));
        add_option<bool>(
            "reopen_closed", "reopen closed nodes if a cheaper path is found",
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
        return plugins::make_shared_from_arg_tuples<
            rectangle_search::RectangleSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"), opts.get<double>("aspect"),
            opts.get<bool>("reopen_closed"), opts.get<bool>("anytime"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<RectangleSearchFeature> _plugin;
}
