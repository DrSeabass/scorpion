#include "boosted_triangle_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_boosted_triangle {
class BoostedTriangleSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, boosted_triangle_search::BoostedTriangleSearch> {
public:
    BoostedTriangleSearchFeature() : TypedFeature("boosted_triangle") {
        document_title("Boosted triangle search (ICAPS-27, scaffold)");
        document_synopsis(
            "Sibling of multi_triangle_search that will grow progress-credit "
            "boosting, heuristic-selection granularity, slope adjustment, and "
            "helpful-action filtering (see icaps-27-plan.md); this revision "
            "is a scaffold clone, bit-identical to multi_triangle_search. "
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
}
