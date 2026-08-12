#include "lazy_triangle_lama_mimick_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_lazy_triangle_lama_mimick {
class LazyTriangleLamaMimickSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, lazy_triangle_lama_mimick_search::LazyTriangleLamaMimickSearch> {
public:
    LazyTriangleLamaMimickSearchFeature() : TypedFeature("lazy_triangle_lama_mimick") {
        document_title("Lazy triangle-LAMA-mimick search");
        document_synopsis(
            "Lazy sibling of triangle_lama_mimick_search, built on this "
            "branch's lazy mechanics (see lazy_boosted_triangle_search): "
            "successors are ranked by their parent's already-known h and "
            "are only evaluated for real (with all N guidance heuristics) "
            "when popped for expansion. "
            "Selection is unchanged from the eager sibling: every list -- "
            "each guidance heuristic's list and each helpful/preferred-only "
            "list -- carries one persistent, global priority counter "
            "(AlternationOpenList's own 'priorities'). The served list for "
            "a whole cascade dive is decided once per step(), before the "
            "cascade loop: the lowest-counter list wins, ties keep the "
            "lowest index. Every expansion the served list actually serves "
            "costs it one point. "
            "The boost (LAMA's actual mechanism): whenever a state's real "
            "evaluation -- now happening at pop time, matching real LAMA's "
            "own LazySearch::step() exactly, rather than the eager "
            "sibling's generation-time approximation -- reports a new "
            "global-best value for any progress-tracked evaluator, every "
            "helpful/preferred-only list's counter drops by 'boost_amount'. "
            "Never fires for the initial state's own evaluation, matching "
            "real LAMA. "
            "preferred_evals=[] (default) leaves no helpful lists to ever "
            "boost, reducing to round-robin-by-priority over the guidance "
            "lists; at a single evaluator this reduces exactly to "
            "lazy_triangle(eval=evals[0], slope=slope). No pruner queue "
            "(see lazy_boosted_triangle_search's own docs for why this lazy "
            "family never grows a guide_by_pruning option).");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "guidance evaluator(s), one ranked list per layer");
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
        add_option<int>(
            "boost_amount",
            "LAMA's boost_preferred magnitude (see alternation_open_list.cc "
            "and the DEFAULT_LAZY_BOOST used by lazy_greedy/lazy_wastar): "
            "every helpful/preferred-only list's priority counter drops by "
            "this much whenever a state's real evaluation (at pop time) "
            "reports a new global-best value for any boosting-eligible "
            "evaluator. 0 makes boosting inert without removing the "
            "helpful lists themselves.",
            "1000");
        add_list_option<shared_ptr<Evaluator>>(
            "preferred_evals",
            "subset of 'evals' (matched by identity) whose preferred "
            "operators each get a paired helpful list per layer -- "
            "exactly LAMA's preferred-only queues, and the only lists "
            "boost_amount ever touches. Empty (default) adds no helpful "
            "lists, making boosting inert.",
            "[]");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "lazy_triangle_lama_mimick");
    }

    virtual shared_ptr<lazy_triangle_lama_mimick_search::LazyTriangleLamaMimickSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<lazy_triangle_lama_mimick_search::LazyTriangleLamaMimickSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<int>("boost_amount"),
            opts.get_list<shared_ptr<Evaluator>>("preferred_evals"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyTriangleLamaMimickSearchFeature> _plugin;
}
