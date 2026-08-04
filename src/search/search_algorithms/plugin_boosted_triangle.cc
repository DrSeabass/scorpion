#include "boosted_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_boosted_triangle {
class BoostedTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, boosted_triangle_search::BoostedTriangleSearch> {
public:
    BoostedTriangleSearchFeature() : TypedFeature("boosted_triangle") {
        document_title("Boosted triangle search (ICAPS-27)");
        document_synopsis(
            "Sibling of multi_triangle_search growing progress-credit "
            "boosting, heuristic-selection granularity, slope adjustment, and "
            "helpful-action filtering (see icaps-27-plan.md). At "
            "credit_boost=0 (default) this is bit-identical to "
            "multi_triangle_search regardless of selection_granularity or "
            "credit_scope; a nonzero credit_boost engages a LAMA-style "
            "per-list token budget, tracked either per depth layer or as a "
            "single budget shared across the search (credit_scope), that "
            "reselects the served list either at each layer boundary or "
            "once per dive (selection_granularity). combo 1b,2c "
            "(credit_scope=global with selection_granularity=per_layer) is "
            "rejected at construction as not meaningful. "
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
            "so successive expansions down a dive alternate heuristics). With a "
            "single evaluator the search reduces to vanilla triangle under "
            "either schedule. Stops after the first plan is found unless "
            "anytime=true.");

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
        add_option<boosted_triangle_search::Schedule>(
            "schedule",
            "round-robin granularity for choosing which guidance list to pop",
            "sweep");
        add_option<boosted_triangle_search::SelectionGranularity>(
            "selection_granularity",
            "ICAPS-27 axis 2 (see icaps-27-plan.md): how often the "
            "credit-driven selector reselects the served list, when "
            "credit_boost != 0 (per_layer: at each layer boundary, from "
            "that layer's own budgets -- requires credit_scope=per_layer; "
            "per_sweep: once per dive, from credit_scope's budget, summed "
            "across active layers if credit_scope=per_layer or read "
            "directly if credit_scope=global). Orthogonal to 'schedule' "
            "and to slope adjustment. At credit_boost == 0 (default) this "
            "option has no effect -- both values reduce to the plain "
            "schedule round-robin.",
            "per_sweep");
        add_option<boosted_triangle_search::CreditScope>(
            "credit_scope",
            "ICAPS-27 axis 1 (see icaps-27-plan.md): whether progress "
            "credit (credit_boost) is tracked per depth layer (default, "
            "combos 1a,2c / 1a,2d) or as a single budget shared across the "
            "whole search (combo 1b,2d only -- credit_scope=global with "
            "selection_granularity=per_layer is combo 1b,2c, which is "
            "rejected at construction as not meaningful).",
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
            "(unclamped). The highest-budget list is served, at the "
            "granularity set by selection_granularity, ties broken by the "
            "schedule round-robin. credit_boost == 0 (default) makes the "
            "whole mechanism inert -- no tokens earned or spent, selection "
            "reduces exactly to the schedule round-robin regardless of "
            "selection_granularity or credit_scope.",
            "0");
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
        add_search_algorithm_options_to_feature(*this, "boosted_triangle");
    }

    virtual shared_ptr<boosted_triangle_search::BoostedTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<boosted_triangle_search::BoostedTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<boosted_triangle_search::Schedule>("schedule"),
            opts.get<boosted_triangle_search::SelectionGranularity>("selection_granularity"),
            opts.get<boosted_triangle_search::CreditScope>("credit_scope"),
            opts.get<int>("credit_boost"),
            opts.get<bool>("guide_by_pruning"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<BoostedTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<boosted_triangle_search::Schedule> _enum_plugin(
    {{"sweep",
      "one guidance list owns the entire cascade dive each step; the served "
      "index rotates between steps (dive-coherent)"},
     {"pop",
      "the served index advances per expansion, so successive expansions down "
      "a dive alternate heuristics (alternation at expansion granularity)"}});

static plugins::TypedEnumPlugin<boosted_triangle_search::SelectionGranularity> _selection_granularity_enum_plugin(
    {{"per_layer",
      "reselect the served list at each layer boundary, from that layer's "
      "own credit_boost token budget (no effect at credit_boost == 0; "
      "requires credit_scope=per_layer)"},
     {"per_sweep",
      "lock the served list for the whole dive, chosen once from "
      "credit_scope's budget (no effect at credit_boost == 0)"}});

static plugins::TypedEnumPlugin<boosted_triangle_search::CreditScope> _credit_scope_enum_plugin(
    {{"per_layer",
      "one independent token budget per depth layer"},
     {"global",
      "one token budget shared across the whole search"}});
}
