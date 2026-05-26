#include "adaptive_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_adaptive_triangle {
class AdaptiveTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, adaptive_triangle_search::AdaptiveTriangleSearch> {
public:
    AdaptiveTriangleSearchFeature() : TypedFeature("adaptive_triangle") {
        document_title("Adaptive triangle search");
        document_synopsis(
            "Triangle search with cascade depth chosen per step from a "
            "heuristic-trend budget. The cascade starts with budget=1; each "
            "frontier extension costs one unit, and each informed "
            "layer-transition (h decreases relative to the previous expansion "
            "in this step) refunds one unit while each uninformed transition "
            "debits one. The cascade halts when the budget is too small to "
            "afford the next extension. Stops after the first plan is found "
            "unless anytime=true.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
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
        add_search_algorithm_options_to_feature(*this, "adaptive_triangle");
    }

    virtual shared_ptr<adaptive_triangle_search::AdaptiveTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<adaptive_triangle_search::AdaptiveTriangleSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<AdaptiveTriangleSearchFeature> _plugin;
}
