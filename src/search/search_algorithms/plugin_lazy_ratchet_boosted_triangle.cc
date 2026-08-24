#include "lazy_ratchet_boosted_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_lazy_ratchet_boosted_triangle {
class LazyRatchetBoostedTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, lazy_ratchet_boosted_triangle_search::LazyRatchetBoostedTriangleSearch> {
public:
    LazyRatchetBoostedTriangleSearchFeature() : TypedFeature("lazy_ratchet_boosted_triangle") {
        document_title("Lazy ratchet-boosted triangle search (ICAPS-27 lazy evaluation, combo 1a,2c + slope adjustment)");
        document_synopsis(
            "lazy_boosted_triangle with the ratchet_triangle_search slope-"
            "adjustment mechanism ported on as-is and always on, exactly "
            "mirroring ratchet_boosted_triangle_search (see "
            "icaps-27-lazy-eval-design.md and "
            "icaps-27-lazy-eval-implementation-prompt.md). Same lazy "
            "mechanics as lazy_boosted_triangle throughout: parent-h "
            "ranking, real evaluation only at pop time, edges as open-list "
            "entries so the credit and ratchet-trend signals both always "
            "use a freshly-computed real h. slope is a persistent state "
            "variable doubled or halved at the end of every step based on "
            "that step's heuristic-trend balance, independent of which "
            "guidance list served each expansion. At credit_boost=0 "
            "(default) list selection is bit-identical to "
            "lazy_boosted_triangle (slope still evolves). See "
            "lazy_ratchet_boosted_sweep_triangle for the once-per-dive "
            "sibling. "
            "Optional pruner queue: the admissible pruning_heuristic, if "
            "set, stays fully lazy exactly like the guidance heuristics -- "
            "evaluated only once a state is popped for expansion, never "
            "per generated successor -- and skips (without expanding) any "
            "state exceeding the current bound. When guide_by_pruning is "
            "also set, that evaluator also seeds one extra ranked list at "
            "generation time, ranked by the parent's h like the guidance "
            "lists, joining the schedule/credit round-robin. "
            "pruning_heuristic unset (default) is an exact no-op reduction "
            "to the behavior above.");

        add_list_option<shared_ptr<Evaluator>>(
            "evals", "guidance evaluator(s), one ranked list per layer");
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
        add_option<lazy_ratchet_boosted_triangle_search::Schedule>(
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
        add_option<bool>(
            "guide_by_pruning",
            "also rank by the admissible pruning_heuristic in an extra open "
            "list (ranked by parent-h at insertion, like the guidance "
            "lists), joining the round-robin. No effect unless "
            "pruning_heuristic is set.",
            "false");
        add_option<shared_ptr<Evaluator>>(
            "pruning_heuristic",
            "admissible evaluator used for f-pruning states popped for "
            "expansion (g + h(pruning_heuristic) >= bound), evaluated "
            "lazily at pop time exactly like the guidance heuristics (see "
            "the class comment). If unset, only g-based pruning applies.",
            plugins::ArgumentInfo::NO_DEFAULT);
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "lazy_ratchet_boosted_triangle");
    }

    virtual shared_ptr<lazy_ratchet_boosted_triangle_search::LazyRatchetBoostedTriangleSearch> create_component(
        const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<lazy_ratchet_boosted_triangle_search::LazyRatchetBoostedTriangleSearch>(
            opts.get_list<shared_ptr<Evaluator>>("evals"),
            opts.get<int>("slope"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<lazy_ratchet_boosted_triangle_search::Schedule>("schedule"),
            opts.get<int>("credit_boost"),
            opts.get_list<shared_ptr<Evaluator>>("preferred_evals"),
            opts.get<bool>("guide_by_pruning"),
            opts.get<shared_ptr<Evaluator>>("pruning_heuristic", nullptr),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<LazyRatchetBoostedTriangleSearchFeature> _plugin;

static plugins::TypedEnumPlugin<lazy_ratchet_boosted_triangle_search::Schedule> _enum_plugin(
    {{"sweep",
      "one guidance list owns the entire cascade dive each step; the served "
      "index rotates between steps (dive-coherent)"},
     {"pop",
      "the served index advances per expansion, so successive expansions down "
      "a dive alternate heuristics (alternation at expansion granularity)"}});
}
