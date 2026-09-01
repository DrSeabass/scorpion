#include "lazy_boosted_triangle_search.h"

#include "../plugins/plugin.h"
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
            "operators. There is no boosting, adaptive slope, pruning "
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
            "each guidance queue receives a preferred-only copy",
            "[]");
        add_search_pruning_options_to_feature(*this);
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
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyMultiTriangleSearchFeature> _plugin;

}
