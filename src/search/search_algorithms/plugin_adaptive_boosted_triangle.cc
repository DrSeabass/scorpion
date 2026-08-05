#include "adaptive_boosted_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_adaptive_boosted_triangle {
class AdaptiveBoostedTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, adaptive_boosted_triangle_search::AdaptiveBoostedTriangleSearch> {
public:
    AdaptiveBoostedTriangleSearchFeature() : TypedFeature("adaptive_boosted_triangle") {
        document_title("Adaptive-boosted triangle search (ICAPS-27, combo 1a,2c + depth-budget adjustment)");
        document_synopsis(
            "boosted_triangle with the adaptive_triangle_search per-step "
            "depth-budget mechanism ported on as-is and always on (see "
            "icaps-27-plan.md step 5): the cascade has no fixed depth cap "
            "-- it runs until it would need to instantiate a new frontier "
            "layer it can't afford. A persistent depth budget (starting at "
            "1, floored at 1 each step) pays one unit per new frontier "
            "layer; an informed layer-transition (h decreases relative to "
            "the previous expansion this step) refunds one unit, an "
            "uninformed one debits one (the original symmetric +1/-1 rule; "
            "adaptive_triangle_search's non_progress_penalty tunable is not "
            "exposed here, per the rule against adding new knobs to a "
            "must-have mechanism). The first frontier extension of a step "
            "is always free. This trend signal is a single step-scoped "
            "counter over every expansion in the step regardless of which "
            "guidance list served it -- orthogonal to the per-list "
            "progress-credit budgets. At credit_boost=0 (default) list "
            "selection is bit-identical to boosted_triangle; a nonzero "
            "credit_boost engages a LAMA-style per-list token budget, one "
            "independent budget per layer, that reselects the served list "
            "at each layer boundary. See adaptive_boosted_sweep_triangle "
            "for the once-per-dive sibling. Triangle search with N "
            "parallel ranked open lists per depth layer, one per "
            "inadmissible guidance heuristic in 'evals'. A successor is "
            "evaluated by all N heuristics and inserted into all N lists "
            "at its layer, so the lists are N orderings of one shared live "
            "frontier; duplicate detection stays global and stale copies "
            "are drained per list. The optional admissible "
            "pruning_heuristic remains the single bound-pruner across all "
            "lists. Scheduling is round-robin across the N lists, with "
            "granularity set by 'schedule': sweep (one list owns the whole "
            "cascade dive each step, rotating between steps) or pop (the "
            "list advances per expansion, so successive expansions down a "
            "dive alternate heuristics). With a single evaluator and "
            "credit_boost=0 the search reduces to adaptive_triangle. Stops "
            "after the first plan is found unless anytime=true.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "inadmissible guidance evaluators, one ranked list per layer");
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent",
            "false");
        add_option<adaptive_boosted_triangle_search::Schedule>(
            "schedule",
            "round-robin granularity for choosing which guidance list to pop",
            "sweep");
        add_option<int>(
            "credit_boost",
            "ICAPS-27 axis 1a (see icaps-27-plan.md): tokens granted to a "
            "list's (guidance heuristic, or the pruner list if "
            "guide_by_pruning is set -- both compete on equal footing) "
            "per-layer budget when one of its expansions improves on that "
            "same list's own previous expansion h at that layer ('an "
            "informed transition'). Every expansion a list serves spends "
            "one token from its budget (unclamped). The highest-budget "
            "list is served at each layer boundary, ties broken by the "
            "schedule round-robin. credit_boost == 0 (default) makes the "
            "whole mechanism inert -- no tokens earned or spent, selection "
            "reduces exactly to the schedule round-robin.",
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
        add_search_algorithm_options_to_feature(*this, "adaptive_boosted_triangle");
    }

    virtual shared_ptr<adaptive_boosted_triangle_search::AdaptiveBoostedTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<adaptive_boosted_triangle_search::AdaptiveBoostedTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<adaptive_boosted_triangle_search::Schedule>("schedule"),
            opts.get<int>("credit_boost"),
            opts.get<bool>("guide_by_pruning"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<AdaptiveBoostedTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<adaptive_boosted_triangle_search::Schedule> _enum_plugin(
    {{"sweep",
      "one guidance list owns the entire cascade dive each step; the served "
      "index rotates between steps (dive-coherent)"},
     {"pop",
      "the served index advances per expansion, so successive expansions down "
      "a dive alternate heuristics (alternation at expansion granularity)"}});
}
