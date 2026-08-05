#include "ratchet_boosted_sweep_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_ratchet_boosted_sweep_triangle {
class RatchetBoostedSweepTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, ratchet_boosted_sweep_triangle_search::RatchetBoostedSweepTriangleSearch> {
public:
    RatchetBoostedSweepTriangleSearchFeature() : TypedFeature("ratchet_boosted_sweep_triangle") {
        document_title("Ratchet-boosted-sweep triangle search (ICAPS-27, combos 1a,2d / 1b,2d + slope adjustment)");
        document_synopsis(
            "boosted_sweep_triangle with the ratchet_triangle_search "
            "slope-adjustment mechanism ported on as-is and always on (see "
            "icaps-27-plan.md step 5): slope is a persistent state variable "
            "doubled or halved at the end of every step based on that "
            "step's heuristic-trend balance (strictly more informed than "
            "uninformed layer-transitions -> double; otherwise halve, floor "
            "1), a single step-scoped signal independent of which guidance "
            "list served each expansion and of credit_scope. At "
            "credit_boost=0 (default) list selection is bit-identical to "
            "boosted_sweep_triangle regardless of credit_scope (slope still "
            "evolves); a nonzero credit_boost engages a LAMA-style token "
            "budget -- summed across every layer currently active "
            "(credit_scope=per_layer) or read directly from a single budget "
            "shared across the search (credit_scope=global) -- that picks "
            "the served list once per step(), before the cascade loop, and "
            "holds it for every layer in that dive. See "
            "ratchet_boosted_triangle for the per-layer reselection "
            "sibling. Triangle search with N parallel ranked open lists per "
            "depth layer, one per inadmissible guidance heuristic in "
            "'evals'. A successor is evaluated by all N heuristics and "
            "inserted into all N lists at its layer, so the lists are N "
            "orderings of one shared live frontier; duplicate detection "
            "stays global and stale copies are drained per list. The "
            "optional admissible pruning_heuristic remains the single "
            "bound-pruner across all lists. Scheduling is round-robin "
            "across the N lists, with granularity set by 'schedule': sweep "
            "(one list owns the whole cascade dive each step, rotating "
            "between steps) or pop (the list advances per expansion, so "
            "successive expansions down a dive alternate heuristics) -- "
            "used as the round-robin fallback/tie-break once credit "
            "engages. With a single evaluator and credit_boost=0 the search "
            "reduces to ratchet_triangle. Stops after the first plan is "
            "found unless anytime=true.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "inadmissible guidance evaluators, one ranked list per layer");
        add_option<int>(
            "slope",
            "initial slope at the start of search; evolves by doubling/halving each step",
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
        add_option<ratchet_boosted_sweep_triangle_search::Schedule>(
            "schedule",
            "round-robin granularity for choosing which guidance list to pop "
            "when credit_boost == 0; otherwise only the round-robin tie-break",
            "sweep");
        add_option<ratchet_boosted_sweep_triangle_search::CreditScope>(
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
        add_search_algorithm_options_to_feature(*this, "ratchet_boosted_sweep_triangle");
    }

    virtual shared_ptr<ratchet_boosted_sweep_triangle_search::RatchetBoostedSweepTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<ratchet_boosted_sweep_triangle_search::RatchetBoostedSweepTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<ratchet_boosted_sweep_triangle_search::Schedule>("schedule"),
            opts.get<ratchet_boosted_sweep_triangle_search::CreditScope>("credit_scope"),
            opts.get<int>("credit_boost"),
            opts.get<bool>("guide_by_pruning"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<RatchetBoostedSweepTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<ratchet_boosted_sweep_triangle_search::Schedule> _enum_plugin(
    {{"sweep",
      "one guidance list owns the entire cascade dive each step; the served "
      "index rotates between steps (dive-coherent)"},
     {"pop",
      "the served index advances per expansion, so successive expansions down "
      "a dive alternate heuristics (alternation at expansion granularity)"}});

static plugins::TypedEnumPlugin<ratchet_boosted_sweep_triangle_search::CreditScope> _credit_scope_enum_plugin(
    {{"per_layer",
      "budgets tracked per depth layer, summed across active layers for "
      "the once-per-dive decision (combo 1a,2d)"},
     {"global",
      "one token budget shared across the whole search, read directly "
      "(combo 1b,2d)"}});
}
