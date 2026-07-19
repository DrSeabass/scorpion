#include "adaptive_rectangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_adaptive_rectangle {
class AdaptiveRectangleSearchFeature
    : public plugins::TypedFeature<
          SearchAlgorithm, adaptive_rectangle_search::AdaptiveRectangleSearch> {
public:
    AdaptiveRectangleSearchFeature() : TypedFeature("adaptive_rectangle") {
        document_title("Adaptive rectangle search");
        document_synopsis(
            "A fork of rectangle search (Lemons, Ruml, Holte, and Sturtevant, "
            "AAAI 2024) whose single parameter -- the aspect ratio -- is set "
            "dynamically rather than fixed. There is no beam width and no "
            "static aspect: the rectangle rotates as the aspect changes. The "
            "aspect is driven by a parameter-free heuristic-trend ratchet (the "
            "same informed/uninformed signal adaptive_triangle uses for its "
            "depth budget) applied to a persistent aspect state. Over each "
            "completed rectangle sweep the search rotates deeper (aspect *= 2) "
            "when informed transitions strictly dominate, wider (aspect /= 2) "
            "when uninformed strictly dominate, and holds otherwise. "
            "adaptive_triangle is the width-1 relative of this search.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<bool>(
            "reopen_closed", "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent "
            "(converges to the optimal cost); the dynamic aspect is designed "
            "for this mode",
            "true");
        add_option<bool>(
            "log_aspect",
            "append one CSV row per completed sweep (expansions,aspect) to "
            "adaptive_rectangle_aspect.csv in the working directory, tracing "
            "the ratcheted aspect ratio. Off => no file is written.",
            "false");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "adaptive_rectangle");
    }

    virtual shared_ptr<adaptive_rectangle_search::AdaptiveRectangleSearch>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            adaptive_rectangle_search::AdaptiveRectangleSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<bool>("reopen_closed"), opts.get<bool>("anytime"),
            opts.get<bool>("log_aspect"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<AdaptiveRectangleSearchFeature> _plugin;
}
