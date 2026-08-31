#include "round_robin_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_round_robin_triangle {
class RoundRobinTriangleSearchFeature
    : public plugins::TypedFeature<
          SearchAlgorithm,
          round_robin_triangle_search::RoundRobinTriangleSearch> {
public:
    RoundRobinTriangleSearchFeature() : TypedFeature("round_robin_triangle") {
        document_title("Per-depth round-robin triangle search");
        document_synopsis(
            "Eager, fixed-slope, first-solution triangle search with one "
            "queue per evaluator at every depth. Each depth has an independent "
            "round-robin cursor. Its cursor advances only when that depth "
            "supplies a live node for expansion; empty and stale-only queue "
            "visits do not rotate it. Intended as a narrow satisficing-search "
            "experimental baseline; it has no preferred queues, boosting, "
            "adaptive slope, anytime mode, or pruning heuristic.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "eager guidance evaluators, one queue per depth each");
        add_option<int>(
            "slope",
            "number of new depth levels added per triangle iteration",
            "1",
            plugins::Bounds("1", "infinity"));
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found",
            "true");
        add_search_algorithm_options_to_feature(*this, "round_robin_triangle");
    }

    virtual shared_ptr<round_robin_triangle_search::RoundRobinTriangleSearch>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            round_robin_triangle_search::RoundRobinTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<RoundRobinTriangleSearchFeature> _plugin;
}
