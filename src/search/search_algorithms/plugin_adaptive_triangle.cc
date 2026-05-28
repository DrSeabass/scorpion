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
            "unless anytime=true. With lift_floor=true (Direction B, relaxed "
            "cascade start-depth) each step's cascade begins at the previous "
            "step's realized dive depth (number of new frontier layers it "
            "instantiated, minus one) below the shallowest active layer "
            "instead of at the root, skipping the shallow layers the last "
            "dive trusted. floor_proxy selects how the start-depth is derived: "
            "layers_added (the previous step's realized dive depth, "
            "conservative) or informedness (position the floor between root "
            "and frontier by the fraction of recent transitions that improved "
            "h, counts reset on each incumbent improvement).");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent",
            "false");
        add_option<bool>(
            "lift_floor",
            "Direction B: start each cascade above the root instead of at "
            "index 0, by the floor_proxy rule, skipping shallow layers the "
            "recent search trusts",
            "false");
        add_option<adaptive_triangle_search::FloorProxy>(
            "floor_proxy",
            "how the lifted cascade start-depth is derived (only used when "
            "lift_floor=true)",
            "informedness");
        add_option<int>(
            "non_progress_penalty",
            "budget decrement applied per uninformed (h non-improving) "
            "transition. Default 1 matches the symmetric +1/-1 budget rule. "
            "Setting to 0 lets the cascade keep running through non-improving "
            "transitions as long as no frontier extension is needed",
            "1",
            plugins::Bounds("0", "infinity"));
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
            opts.get<bool>("lift_floor"),
            opts.get<adaptive_triangle_search::FloorProxy>("floor_proxy"),
            opts.get<int>("non_progress_penalty"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<AdaptiveTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<adaptive_triangle_search::FloorProxy> _enum_plugin(
    {{"layers_added",
      "previous step's realized dive depth (number of new frontier layers it "
      "instantiated); conservative"},
     {"informedness",
      "position the floor between root and frontier by the fraction of recent "
      "h-improving transitions; counts reset on each incumbent improvement"}});
}
