#include "round_robin_triangle_search.h"

#include "../plugins/plugin.h"
#include "../utils/rng_options.h"

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
            "experimental baseline. schedule=sweep locks one queue for a "
            "complete dive; schedule=depth gives each depth an independent "
            "cursor. Optional preferred queues use the union of all preferred "
            "evaluators. With global_preferred=true and schedule=sweep, preferred "
            "successors instead share one global FIFO queue. There is no "
            "boosting, adaptive slope, anytime mode, "
            "or pruning heuristic.");

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
        add_option<round_robin_triangle_search::Schedule>(
            "schedule", "queue scheduling policy", "depth");
        add_list_option<shared_ptr<Evaluator>>(
            "preferred_evals",
            "empty or all guidance evaluators; their preferred operators are "
            "unioned into a preferred-only copy of each guidance queue, or "
            "one shared FIFO queue when global_preferred=true",
            "[]");
        add_option<bool>(
            "global_preferred",
            "use one FIFO preferred queue across all depths; requires "
            "schedule=sweep and nonempty preferred_evals. Its sweep gets the "
            "usual expansion budget, with successors inserted at their actual depth",
            "false");
        add_option<int>(
            "k", "maximum live expansions per depth visit from the selected queue",
            "1", plugins::Bounds("1", "infinity"));
        add_option<bool>(
            "expand_equal",
            "expand the entire minimum (h,g) group from the selected depth queue; "
            "requires k=1. Only depth-stratified queues are batched",
            "false");
        add_option<bool>(
            "random_start", "start each sweep at a uniformly random absolute depth "
            "between the root and the deepest queued layer, inclusive", "false");
        add_option<int>(
            "window", "start each sweep this many depth layers behind the deepest "
            "queued layer; 0 starts at the front, -1 disables the window. "
            "Cannot be combined with random_start", "-1",
            plugins::Bounds("-1", "infinity"));
        utils::add_rng_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "round_robin_triangle");
    }

    virtual shared_ptr<round_robin_triangle_search::RoundRobinTriangleSearch>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            round_robin_triangle_search::RoundRobinTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<round_robin_triangle_search::Schedule>("schedule"),
            opts.get_list<shared_ptr<Evaluator>>("preferred_evals"),
            get_search_algorithm_arguments_from_options(opts),
            opts.get<bool>("global_preferred"),
            opts.get<int>("k"),
            opts.get<bool>("expand_equal"),
            opts.get<bool>("random_start"),
            utils::get_rng_arguments_from_options(opts),
            opts.get<int>("window"));
    }
};

static plugins::FeaturePlugin<RoundRobinTriangleSearchFeature> _plugin;
static plugins::TypedEnumPlugin<round_robin_triangle_search::Schedule> _enum_plugin(
    {{"sweep", "one queue owns the complete cascade dive"},
     {"depth", "each depth owns an independent round-robin cursor"}});
}
