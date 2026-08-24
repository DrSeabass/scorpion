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
            "lazy_triangle(eval=evals[0], slope=slope). "
            "Optional pruner queue: the admissible pruning_heuristic, if "
            "set, stays fully lazy exactly like the guidance heuristics -- "
            "evaluated only once a state is popped for expansion, never "
            "per generated successor -- and skips (without expanding) any "
            "state exceeding the current bound. When guide_by_pruning is "
            "also set, that evaluator also seeds one extra ranked list at "
            "generation time (ranked by the parent's h like the guidance "
            "lists) that participates in the priority-counter round-robin "
            "like any guidance list (never boosted by boost_amount, which "
            "stays scoped to the helpful/preferred-only lists). "
            "pruning_heuristic unset (default) is an exact no-op reduction "
            "to the behavior above.");

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
            "evaluators whose preferred operators are unioned. When nonempty, "
            "every guidance queue gets an interleaved preferred-only copy; "
            "these are the only lists "
            "boost_amount ever touches. Empty (default) adds no helpful "
            "lists, making boosting inert.",
            "[]");
        add_option<bool>(
            "guide_by_pruning",
            "also rank by the admissible pruning_heuristic in an extra open "
            "list (ranked by parent-h at insertion, like the guidance "
            "lists), joining the priority-counter round-robin. No effect "
            "unless pruning_heuristic is set.",
            "false");
        add_option<shared_ptr<Evaluator>>(
            "pruning_heuristic",
            "admissible evaluator used for f-pruning states popped for "
            "expansion (g + h(pruning_heuristic) >= bound), evaluated "
            "lazily at pop time exactly like the guidance heuristics (see "
            "the class comment). If unset, only g-based pruning applies.",
            plugins::ArgumentInfo::NO_DEFAULT);
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
            opts.get<bool>("guide_by_pruning"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyTriangleLamaMimickSearchFeature> _plugin;
}
