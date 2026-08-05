#include "boosted_sweep_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_boosted_sweep_triangle {
class BoostedSweepTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, boosted_sweep_triangle_search::BoostedSweepTriangleSearch> {
public:
    BoostedSweepTriangleSearchFeature() : TypedFeature("boosted_sweep_triangle") {
        document_title("Boosted-sweep triangle search (ICAPS-27, combos 1a,2d / 1b,2d)");
        document_synopsis(
            "Sibling of boosted_triangle with the served list decided once "
            "per dive instead of at every layer boundary (see "
            "icaps-27-plan.md, combos 1a,2d / 1b,2d). At credit_boost=0 "
            "(default) this is bit-identical to multi_triangle_search "
            "regardless of credit_scope; a nonzero credit_boost engages a "
            "LAMA-style token budget -- summed across every layer currently "
            "active (credit_scope=per_layer) or read directly from a single "
            "budget shared across the search (credit_scope=global) -- that "
            "picks the served list once per step(), before the cascade "
            "loop, and holds it for every layer in that dive. "
            "Triangle search with N parallel ranked open lists per depth "
            "layer, one per inadmissible guidance heuristic in 'evals'. A "
            "successor is evaluated by all N heuristics and inserted into all "
            "N lists at its layer, so the lists are N orderings of one shared "
            "live frontier; duplicate detection stays global and stale copies "
            "are drained per list. The optional admissible pruning_heuristic "
            "remains the single bound-pruner across all lists. Scheduling is "
            "round-robin across the N lists, with granularity set by "
            "'schedule': sweep (one list owns the whole cascade dive each step, "
            "rotating between steps) or pop (the list advances per expansion, "
            "so successive expansions down a dive alternate heuristics) -- "
            "used as the round-robin fallback/tie-break once credit engages. "
            "With a single evaluator the search reduces to vanilla triangle. "
            "Stops after the first plan is found unless anytime=true. "
            "ICAPS-27 step 6 (helpful actions): each evaluator named in "
            "'preferred_evals' (a subset of 'evals', matched by identity) "
            "gets one extra 'helpful' list per layer, ranked by that same "
            "evaluator's h but populated only with successors reached via "
            "one of that evaluator's own preferred operators on the parent "
            "-- unlike the guidance/pruner lists, a helpful list holds a "
            "sparse subset of the layer's live content. Helpful lists "
            "compete on equal footing with the guidance and pruner lists. "
            "Preferred-operator sets are computed once per state (when it "
            "is first evaluated as a successor) and reused at expansion "
            "time, so no evaluator is ever run twice just to learn its "
            "preferred operators. preferred_evals=[] (default) adds no "
            "helpful lists and is an exact no-op reduction to the "
            "pre-step-6 behavior described above.");

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
        add_option<boosted_sweep_triangle_search::Schedule>(
            "schedule",
            "round-robin granularity for choosing which guidance list to pop "
            "when credit_boost == 0; otherwise only the round-robin tie-break",
            "sweep");
        add_option<boosted_sweep_triangle_search::CreditScope>(
            "credit_scope",
            "ICAPS-27 axis 1 (see icaps-27-plan.md): whether progress "
            "credit (credit_boost) is tracked per depth layer, summed "
            "across active layers for the once-per-dive decision (combo "
            "1a,2d), or as a single budget shared across the whole search, "
            "read directly (combo 1b,2d).",
            "per_layer");
        add_option<int>(
            "credit_boost",
            "ICAPS-27 axis 1 (see icaps-27-plan.md): tokens granted to a "
            "list's (guidance heuristic, or the pruner list if "
            "guide_by_pruning is set -- both compete on equal footing) "
            "budget when one of its expansions improves on that same "
            "list's own previous expansion h ('an informed transition'; "
            "scoped per layer or globally, see credit_scope). Every "
            "expansion a list serves spends one token from its budget "
            "(unclamped). The highest-budget list is served for the whole "
            "dive, ties broken by the schedule round-robin. credit_boost "
            "== 0 (default) makes the whole mechanism inert -- no tokens "
            "earned or spent, selection reduces exactly to the schedule "
            "round-robin regardless of credit_scope.",
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
        add_option<bool>(
            "guide_by_pruning",
            "also rank by the admissible pruning_heuristic in an extra open "
            "list, joining the round-robin. The admissible h is already "
            "computed for the f-prune, so this adds list memory, not "
            "evaluation; it does force the prune eval to be unconditional. "
            "No effect unless pruning_heuristic is set.",
            "false");
        add_option<shared_ptr<Evaluator>>(
            "pruning_heuristic",
            "admissible evaluator used for f-pruning successors "
            "(g + h(pruning_heuristic) >= bound). "
            "If unset, only g-based pruning applies.",
            plugins::ArgumentInfo::NO_DEFAULT);
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "boosted_sweep_triangle");
    }

    virtual shared_ptr<boosted_sweep_triangle_search::BoostedSweepTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<boosted_sweep_triangle_search::BoostedSweepTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<boosted_sweep_triangle_search::Schedule>("schedule"),
            opts.get<boosted_sweep_triangle_search::CreditScope>("credit_scope"),
            opts.get<int>("credit_boost"),
            opts.get_list<shared_ptr<Evaluator>>("preferred_evals"),
            opts.get<bool>("guide_by_pruning"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<BoostedSweepTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<boosted_sweep_triangle_search::Schedule> _enum_plugin(
    {{"sweep",
      "one guidance list owns the entire cascade dive each step; the served "
      "index rotates between steps (dive-coherent)"},
     {"pop",
      "the served index advances per expansion, so successive expansions down "
      "a dive alternate heuristics (alternation at expansion granularity)"}});

static plugins::TypedEnumPlugin<boosted_sweep_triangle_search::CreditScope> _credit_scope_enum_plugin(
    {{"per_layer",
      "budgets tracked per depth layer, summed across active layers for "
      "the once-per-dive decision (combo 1a,2d)"},
     {"global",
      "one token budget shared across the whole search, read directly "
      "(combo 1b,2d)"}});
}
