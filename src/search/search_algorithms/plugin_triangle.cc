#include "triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_triangle {
class TriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, triangle_search::TriangleSearch> {
public:
    TriangleSearchFeature() : TypedFeature("triangle") {
        document_title("Triangle search");
        document_synopsis(
            "Triangle search with one ranked open list per depth layer. "
            "Stops after the first plan is found unless anytime=true.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<int>(
            "slope",
            "number of new depth levels added per triangle iteration",
            "1",
            plugins::Bounds("1", "infinity"));
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent",
            "false");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "triangle");
    }

    virtual shared_ptr<triangle_search::TriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<triangle_search::TriangleSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<TriangleSearchFeature> _plugin;
}
