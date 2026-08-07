#include "lazy_boosted_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_lazy_boosted_triangle {
class LazyBoostedTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, lazy_boosted_triangle_search::LazyBoostedTriangleSearch> {
public:
    LazyBoostedTriangleSearchFeature() : TypedFeature("lazy_boosted_triangle") {
        document_title("Lazy boosted triangle search (ICAPS-27 lazy evaluation)");
        document_synopsis(
            "Lazy sibling of boosted_triangle_search (see "
            "icaps-27-lazy-eval-design.md and "
            "icaps-27-lazy-eval-implementation-prompt.md). Successors are "
            "ranked by their parent's already-known h -- reused for free via "
            "EvaluationContext's copy constructor, exactly as "
            "lazy_triangle_search and FD's own LazySearch do it -- and are "
            "only evaluated for real (with all N guidance heuristics) when "
            "popped for expansion, so evaluation cost is paid only for "
            "states that are actually expanded, never for states that are "
            "merely generated. Every successor is inserted into all N "
            "guidance lists per layer, each ranked by that evaluator's own "
            "parent-h. "
            "ICAPS-27 progress credit (axis 1a, per-layer) and selection "
            "(axis 2c, per-layer reselection -- matching boosted_triangle_"
            "search's own combo, see icaps-27-plan.md) are both implemented: "
            "a list's per-layer budget earns credit_boost tokens on an "
            "informed transition (its own h improving relative to its own "
            "previous expansion at that layer) and spends one token per "
            "expansion it serves; the highest-budget non-empty list is "
            "served at each layer boundary, ties and credit_boost=0 (the "
            "default) falling back to plain round-robin ('schedule'). "
            "Because open-list entries here are edges, not evaluated "
            "{id,h,g} triples, the credit signal always uses the state's "
            "freshly-computed real h from the same pop that is expanding "
            "it -- there is no stored-h field that could go stale. "
            "ICAPS-27 step 6 (helpful actions): each evaluator named in "
            "'preferred_evals' (a subset of 'evals', matched by identity) "
            "gets one extra 'helpful' list per layer, ranked by that same "
            "evaluator's parent-h but populated only with successors "
            "reached via one of that evaluator's own preferred operators "
            "on the parent. Unlike the eager sibling, preferred-operator "
            "sets are not cached: they are computed once, in the same call "
            "that evaluates a state for real (at pop time), which is also "
            "the only moment they are ever needed. preferred_evals=[] "
            "(default) adds no helpful lists and is an exact no-op "
            "reduction to the behavior above. "
            "At num_lists == 1 and preferred_evals == [] (any credit_boost, "
            "any schedule) this reduces exactly to "
            "lazy_triangle(eval=evals[0], slope=slope). No pruner queue.");

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
        add_option<lazy_boosted_triangle_search::Schedule>(
            "schedule",
            "round-robin granularity for choosing which guidance list to pop",
            "sweep");
        add_option<int>(
            "credit_boost",
            "ICAPS-27 axis 1a (see icaps-27-plan.md): tokens granted to a "
            "list's per-layer budget when one of its expansions improves on "
            "that same list's own previous expansion h at that layer ('an "
            "informed transition'). Every expansion a list serves spends "
            "one token from its budget (unclamped). The highest-budget "
            "list is served at each layer boundary, ties broken by the "
            "schedule round-robin. credit_boost == 0 (default) makes the "
            "whole mechanism inert -- no tokens earned or spent, selection "
            "reduces exactly to the schedule round-robin.",
            "0");
        add_list_option<shared_ptr<Evaluator>>(
            "preferred_evals",
            "ICAPS-27 step 6 (see icaps-27-plan.md): subset of 'evals' "
            "(matched by identity) whose preferred operators each get a "
            "paired helpful list per layer, populated only with successors "
            "reached via that evaluator's own preferred operator on the "
            "parent. Empty (default) adds no helpful lists and is an exact "
            "no-op reduction to the pre-step-6 behavior.",
            "[]");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "lazy_boosted_triangle");
    }

    virtual shared_ptr<lazy_boosted_triangle_search::LazyBoostedTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<lazy_boosted_triangle_search::LazyBoostedTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<lazy_boosted_triangle_search::Schedule>("schedule"),
            opts.get<int>("credit_boost"),
            opts.get_list<shared_ptr<Evaluator>>("preferred_evals"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyBoostedTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<lazy_boosted_triangle_search::Schedule> _enum_plugin(
    {{"sweep",
      "one guidance list owns the entire cascade dive each step; the served "
      "index rotates between steps (dive-coherent)"},
     {"pop",
      "the served index advances per expansion, so successive expansions down "
      "a dive alternate heuristics (alternation at expansion granularity)"}});
}
