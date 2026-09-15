#include "lazy_boosted_triangle_search.h"

#include "../plugins/plugin.h"
#include "../utils/rng_options.h"
#include "../utils/system.h"

using namespace std;

namespace plugin_lazy_multi_triangle {

class LazyMultiTriangleSearchFeature
    : public plugins::TypedFeature<
          SearchAlgorithm,
          lazy_boosted_triangle_search::LazyBoostedTriangleSearch> {
public:
    LazyMultiTriangleSearchFeature() : TypedFeature("lazy_multi_triangle") {
        document_title("Lazy multi-heuristic triangle search");
        document_synopsis(
            "Clean fixed-slope, first-solution lazy multi-heuristic triangle "
            "search. Successors are ranked by their parent's known heuristic "
            "values and evaluated for real only when popped. schedule=sweep "
            "locks one queue for the complete dive and skips depths where "
            "that queue is empty. schedule=depth gives every depth an "
            "independent persistent round-robin cursor; empty and stale-only "
            "queues are scanned past, and the cursor advances only after a "
            "live node is expanded. When preferred_evals is nonempty, every "
            "guidance queue gets a preferred-only copy and all preferred "
            "evaluators contribute to one LAMA-style union of preferred "
            "operators. With global_preferred=true and schedule=sweep, preferred "
            "successors instead share one global FIFO queue. There is no "
            "boosting, adaptive slope, pruning "
            "heuristic, or anytime mode.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "guidance evaluators, one ordinary queue per depth each");
        add_option<int>(
            "slope",
            "number of new depth levels added per triangle iteration",
            "1",
            plugins::Bounds("1", "infinity"));
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<lazy_boosted_triangle_search::Schedule>(
            "schedule",
            "queue scheduling policy; this algorithm supports sweep and depth",
            "sweep");
        add_list_option<shared_ptr<Evaluator>>(
            "preferred_evals",
            "evaluators whose preferred operators are unioned; when nonempty, "
            "each guidance queue receives a preferred-only copy unless "
            "global_preferred=true selects one shared FIFO queue",
            "[]");
        add_search_pruning_options_to_feature(*this);
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
            "requires k=1. Lazy search uses parent h and insertion-time successor g. "
            "Only depth-stratified queues are batched",
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
        add_search_algorithm_options_to_feature(*this, "lazy_multi_triangle");
    }

    virtual shared_ptr<lazy_boosted_triangle_search::LazyBoostedTriangleSearch>
    create_component(const plugins::Options &opts) const override {
        auto schedule =
            opts.get<lazy_boosted_triangle_search::Schedule>("schedule");
        if (schedule == lazy_boosted_triangle_search::Schedule::POP) {
            cerr << "lazy_multi_triangle supports schedule=sweep or depth, not pop."
                 << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        return plugins::make_shared_from_arg_tuples<
            lazy_boosted_triangle_search::LazyBoostedTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            false,
            schedule,
            0,
            true,
            true,
            opts.get_list<shared_ptr<Evaluator>>("preferred_evals"),
            false,
            nullptr,
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts),
            opts.get<bool>("global_preferred"),
            opts.get<int>("k"),
            opts.get<bool>("expand_equal"),
            opts.get<bool>("random_start"),
            utils::get_rng_arguments_from_options(opts),
            opts.get<int>("window"));
    }
};

static plugins::FeaturePlugin<LazyMultiTriangleSearchFeature> _plugin;

}
