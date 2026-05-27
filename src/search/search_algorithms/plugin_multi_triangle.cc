#include "multi_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_multi_triangle {
class MultiTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, multi_triangle_search::MultiTriangleSearch> {
public:
    MultiTriangleSearchFeature() : TypedFeature("multi_triangle") {
        document_title("Multi-heuristic triangle search");
        document_synopsis(
            "Triangle search with N parallel ranked open lists per depth "
            "layer, one per inadmissible guidance heuristic in 'evals'. A "
            "successor is evaluated by all N heuristics and inserted into all "
            "N lists at its layer, so the lists are N orderings of one shared "
            "live frontier; duplicate detection stays global and stale copies "
            "are drained per list. The optional admissible pruning_heuristic "
            "remains the single bound-pruner across all lists. Scheduling is "
            "per-sweep round-robin: one heuristic list owns the entire cascade "
            "dive each step, and the served index rotates between steps. With "
            "a single evaluator the search reduces to vanilla triangle. Stops "
            "after the first plan is found unless anytime=true.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "inadmissible guidance evaluators, one ranked list per layer");
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
        add_option<shared_ptr<Evaluator>>(
            "pruning_heuristic",
            "admissible evaluator used for f-pruning successors "
            "(g + h(pruning_heuristic) >= bound). "
            "If unset, only g-based pruning applies.",
            plugins::ArgumentInfo::NO_DEFAULT);
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "multi_triangle");
    }

    virtual shared_ptr<multi_triangle_search::MultiTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<multi_triangle_search::MultiTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<MultiTriangleSearchFeature> _plugin;
}
