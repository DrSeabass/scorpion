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
            "selection_granularity=per_sweep (default) this is bit-identical "
            "to multi_triangle_search; per_layer additionally reselects the "
            "served guidance list at each layer boundary via a LAMA-style "
            "per-heuristic token budget (see credit_boost). "
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
            "credit-driven heuristic selector reselects the active guidance "
            "list (per_layer: at each layer boundary, via credit_boost; "
            "per_sweep: locked for the whole dive -- still a plain schedule "
            "round-robin, credit_boost has no effect). Orthogonal to "
            "'schedule' and to slope adjustment.",
            "per_sweep");
        add_option<int>(
            "credit_boost",
            "ICAPS-27 axis 1a (see icaps-27-plan.md): tokens granted to a "
            "list's (guidance heuristic, or the pruner list if "
            "guide_by_pruning is set -- both compete on equal footing) "
            "per-layer budget when one of its expansions improves on that "
            "same list's own previous expansion h at that layer ('an "
            "informed transition'). Every expansion a list serves spends "
            "one token from its budget (unclamped). Only consulted when "
            "selection_granularity == per_layer, where the highest-budget "
            "list is served at each layer boundary (ties broken by the "
            "schedule round-robin). credit_boost == 0 (default) means "
            "tokens are only ever spent, never earned -- a fair-share "
            "(least-served-first) policy, not an exact reduction to the "
            "round-robin.",
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
      "reselect the active guidance list at each layer boundary, via the "
      "credit_boost token budget"},
     {"per_sweep",
      "lock the active guidance list for the whole dive, reselect only "
      "between sweeps (still a plain schedule round-robin)"}});
}
