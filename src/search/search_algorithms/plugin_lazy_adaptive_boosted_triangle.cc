#include "lazy_adaptive_boosted_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_lazy_adaptive_boosted_triangle {
class LazyAdaptiveBoostedTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, lazy_adaptive_boosted_triangle_search::LazyAdaptiveBoostedTriangleSearch> {
public:
    LazyAdaptiveBoostedTriangleSearchFeature() : TypedFeature("lazy_adaptive_boosted_triangle") {
        document_title("Lazy adaptive-boosted triangle search (ICAPS-27 lazy evaluation, combo 1a,2c + depth-budget adjustment)");
        document_synopsis(
            "lazy_boosted_triangle with the adaptive_triangle_search "
            "per-step depth-budget mechanism ported on as-is and always on, "
            "mirroring adaptive_boosted_triangle_search (see "
            "icaps-27-lazy-eval-design.md and "
            "icaps-27-lazy-eval-implementation-prompt.md). The cascade has "
            "no fixed depth cap -- it runs until it would need to "
            "instantiate a new frontier layer it can't afford. A persistent "
            "depth budget (starting at 1, floored at 1 each step) pays one "
            "unit per new frontier layer; an informed layer-transition "
            "refunds one unit, an uninformed one debits one. The first "
            "frontier extension of a step is always free. Same lazy "
            "mechanics as lazy_boosted_triangle throughout: parent-h "
            "ranking, real evaluation only at pop time, edges as open-list "
            "entries. Structural note: EdgeOpenList only supports "
            "destructive remove_min() (no peek), so a budget-rejected edge "
            "is held in a single-slot buffer rather than left in place on "
            "the list, standing in for the eager sibling's non-destructive "
            "peek -- see the .h file for the full discussion. At "
            "credit_boost=0 (default) list selection is bit-identical to "
            "lazy_boosted_triangle; the depth budget always evolves "
            "regardless.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "guidance evaluator(s), one ranked list per layer");
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent",
            "false");
        add_option<lazy_adaptive_boosted_triangle_search::Schedule>(
            "schedule",
            "round-robin granularity for choosing which guidance list to pop",
            "sweep");
        add_option<int>(
            "credit_boost",
            "ICAPS-27 axis 1a (see icaps-27-plan.md): tokens granted to a "
            "list's per-layer budget when one of its expansions improves on "
            "that same list's own previous expansion h at that layer. "
            "credit_boost == 0 (default) makes the mechanism inert.",
            "0");
        add_list_option<shared_ptr<Evaluator>>(
            "preferred_evals",
            "ICAPS-27 step 6 (see icaps-27-plan.md): subset of 'evals' "
            "(matched by identity) whose preferred operators each get a "
            "paired helpful list per layer. Empty (default) adds no "
            "helpful lists and is an exact no-op reduction.",
            "[]");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "lazy_adaptive_boosted_triangle");
    }

    virtual shared_ptr<lazy_adaptive_boosted_triangle_search::LazyAdaptiveBoostedTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<lazy_adaptive_boosted_triangle_search::LazyAdaptiveBoostedTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<lazy_adaptive_boosted_triangle_search::Schedule>("schedule"),
            opts.get<int>("credit_boost"),
            opts.get_list<shared_ptr<Evaluator>>("preferred_evals"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyAdaptiveBoostedTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<lazy_adaptive_boosted_triangle_search::Schedule> _enum_plugin(
    {{"sweep",
      "one guidance list owns the entire cascade dive each step; the served "
      "index rotates between steps (dive-coherent)"},
     {"pop",
      "the served index advances per expansion, so successive expansions down "
      "a dive alternate heuristics (alternation at expansion granularity)"}});
}
