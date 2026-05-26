#include "ratchet_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_ratchet_triangle {
class RatchetTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, ratchet_triangle_search::RatchetTriangleSearch> {
public:
    RatchetTriangleSearchFeature() : TypedFeature("ratchet_triangle") {
        document_title("Ratchet triangle search");
        document_synopsis(
            "Triangle search whose slope is a persistent state variable "
            "doubled or halved at the end of each step based on that step's "
            "heuristic-trend balance. If the step had strictly more informed "
            "layer-transitions (h of the current expansion below h of the "
            "previous expansion in this step) than uninformed ones, slope "
            "doubles; otherwise it halves, with a floor of 1. No upper bound "
            "beyond int-overflow guarding. Stops after the first plan is "
            "found unless anytime=true.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator");
        add_option<int>(
            "slope",
            "initial slope at the start of search; evolves by doubling/halving each step",
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
        add_search_algorithm_options_to_feature(*this, "ratchet_triangle");
    }

    virtual shared_ptr<ratchet_triangle_search::RatchetTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<ratchet_triangle_search::RatchetTriangleSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<RatchetTriangleSearchFeature> _plugin;
}
