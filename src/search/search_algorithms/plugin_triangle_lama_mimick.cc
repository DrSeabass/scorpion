#include "triangle_lama_mimick_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_triangle_lama_mimick {
class TriangleLamaMimickSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, triangle_lama_mimick_search::TriangleLamaMimickSearch> {
public:
    TriangleLamaMimickSearchFeature() : TypedFeature("triangle_lama_mimick") {
        document_title("Triangle-LAMA-mimick search (ICAPS-27)");
        document_synopsis(
            "boosted_sweep_triangle (schedule=sweep) with LAMA's own "
            "alternation-queue policy (see alternation_open_list.cc) "
            "substituted for boosted_sweep_triangle's self-referential "
            "progress-credit mechanism. Every list -- each guidance "
            "heuristic's list, each helpful/preferred-only list, and the "
            "optional pruner list -- carries one persistent, global "
            "priority counter, all starting at 0. The served list for a "
            "whole cascade dive is decided once per step(), before the "
            "cascade loop: the lowest-counter list wins (ties keep the "
            "lowest index), exactly like AlternationOpenList::remove_min. "
            "Every expansion the served list actually serves costs it one "
            "point. With no boosting this alone is round-robin over all "
            "lists in index order.\n"
            "The boost: whenever a newly generated successor's evaluation "
            "reports a new global-best value for any evaluator used for "
            "boosting (every Heuristic, unconditionally), every helpful/"
            "preferred-only list's counter drops by 'boost_amount', "
            "exactly like AlternationOpenList::boost_preferred() -- the "
            "reward always lands on the whole preferred-list category, "
            "never on whichever heuristic happened to cause the progress. "
            "This is LAMA's real trigger and reward, translated to eager "
            "evaluation: LAMA checks/rewards progress when a node is "
            "popped for expansion (its heuristics are lazy, evaluated at "
            "pop time); this algorithm's heuristics are eager, so the "
            "check happens where a successor is first evaluated, at "
            "generation time -- and, matching LAMA, the initial state's "
            "own evaluation never triggers a boost.\n"
            "preferred_evals=[] (default) leaves no helpful lists to ever "
            "boost, so this reduces exactly to round-robin over the "
            "guidance (and optional pruner) lists -- i.e. "
            "multi_triangle_search with schedule=sweep.\n"
            "Each depth layer holds N parallel ranked lists, one per "
            "inadmissible guidance heuristic in 'evals'; a successor is "
            "evaluated by all N heuristics and inserted into all N lists "
            "at its layer, so the N lists are N orderings of one shared "
            "live frontier -- duplicate detection stays global. Each "
            "evaluator named in 'preferred_evals' (a subset of 'evals', "
            "matched by identity) gets one additional 'helpful' list per "
            "layer, ranked by that same evaluator's h but populated only "
            "with successors reached via one of that evaluator's own "
            "preferred operators on the parent. Preferred-operator sets "
            "are computed once per state, when it is first evaluated as a "
            "successor, and reused at expansion time, so no evaluator is "
            "ever run a second time just to learn which of its operators "
            "are preferred. The optional admissible pruning_heuristic "
            "remains the single bound-pruner across all lists; "
            "guide_by_pruning additionally gives it its own ranked list, "
            "competing on equal footing but never boosted. Stops after "
            "the first plan is found unless anytime=true.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "inadmissible guidance evaluators, one ranked list per layer");
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
            "this much whenever a newly generated successor's evaluation "
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
        add_option<bool>(
            "guide_by_pruning",
            "also rank by the admissible pruning_heuristic in an extra "
            "open list, joining the round-robin/boost pool as a plain "
            "(never-boosted) list. The admissible h is already computed "
            "for the f-prune, so this adds list memory, not evaluation; it "
            "does force the prune eval to be unconditional. No effect "
            "unless pruning_heuristic is set.",
            "false");
        add_option<shared_ptr<Evaluator>>(
            "pruning_heuristic",
            "admissible evaluator used for f-pruning successors "
            "(g + h(pruning_heuristic) >= bound). "
            "If unset, only g-based pruning applies.",
            plugins::ArgumentInfo::NO_DEFAULT);
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "triangle_lama_mimick");
    }

    virtual shared_ptr<triangle_lama_mimick_search::TriangleLamaMimickSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<triangle_lama_mimick_search::TriangleLamaMimickSearch>(
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

static plugins::FeaturePlugin<TriangleLamaMimickSearchFeature> _plugin;
}
