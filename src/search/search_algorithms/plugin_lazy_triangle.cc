#include "lazy_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_lazy_triangle {
class LazyTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, lazy_triangle_search::LazyTriangleSearch> {
public:
    LazyTriangleSearchFeature() : TypedFeature("lazy_triangle") {
        document_title("Lazy triangle search");
        document_synopsis(
            "Triangle search with deferred (lazy) successor evaluation. "
            "Keeps one edge open list per depth layer and follows the triangle "
            "expansion schedule controlled by slope.");

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
        add_search_algorithm_options_to_feature(*this, "lazy_triangle");
    }

    virtual shared_ptr<lazy_triangle_search::LazyTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<lazy_triangle_search::LazyTriangleSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyTriangleSearchFeature> _plugin;
}
