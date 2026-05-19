#include "graph_discrepancy_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_gds {
class GraphDiscrepancySearchFeature
    : public plugins::TypedFeature<
          SearchAlgorithm, graph_discrepancy_search::GraphDiscrepancySearch> {
public:
    GraphDiscrepancySearchFeature() : TypedFeature("gds") {
        document_title("Graph discrepancy search");
        document_synopsis(
            "Graph discrepancy search. Children are ranked by 'eval'. With "
            "discrepancy_mode=binary the best-ranked child gets discrepancy 0 and "
            "all others discrepancy 1; with discrepancy_mode=child_rank, discrepancy "
            "is the child rank (0, 1, 2, ...); with discrepancy_mode=h_gap, discrepancy "
            "is h(child)-h(best_child). The open queue is ordered by the path-sum of "
            "discrepancies (TD). With anytime=true (default), the search maintains an "
            "incumbent and prunes states with g + prune_eval >= incumbent cost, continuing "
            "until the open list is exhausted. With anytime=false, search stops on the "
            "first solution found.");

        add_option<shared_ptr<Evaluator>>(
            "eval", "discrepancy ranking evaluator");
        add_option<shared_ptr<Evaluator>>(
            "prune_eval",
            "admissible evaluator for incumbent pruning; defaults to eval if omitted",
            plugins::ArgumentInfo::NO_DEFAULT);
        add_option<bool>(
            "reopen_closed", "reopen closed nodes if a cheaper path is found",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent",
            "true");
        add_option<string>(
            "discrepancy_mode",
            "discrepancy definition: 'binary', 'child_rank', or 'h_gap'",
            "\"binary\"");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "gds");
    }

    virtual shared_ptr<graph_discrepancy_search::GraphDiscrepancySearch>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            graph_discrepancy_search::GraphDiscrepancySearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<shared_ptr<Evaluator>>("prune_eval", nullptr),
            opts.get<bool>("reopen_closed"), opts.get<bool>("anytime"),
            opts.get<string>("discrepancy_mode"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<GraphDiscrepancySearchFeature> _plugin;

class GraphDiscrepancySearchAliasFeature
    : public plugins::TypedFeature<
          SearchAlgorithm, graph_discrepancy_search::GraphDiscrepancySearch> {
public:
    GraphDiscrepancySearchAliasFeature() : TypedFeature("graph_discrepancy") {
        document_title("Graph discrepancy search (alias)");
        document_synopsis("Alias for 'gds'. See 'gds' for full documentation.");

        add_option<shared_ptr<Evaluator>>(
            "eval", "discrepancy ranking evaluator");
        add_option<shared_ptr<Evaluator>>(
            "prune_eval",
            "admissible evaluator for incumbent pruning; defaults to eval if omitted",
            plugins::ArgumentInfo::NO_DEFAULT);
        add_option<bool>("reopen_closed", "reopen closed nodes", "true");
        add_option<bool>(
            "anytime", "continue after finding a solution", "true");
        add_option<string>(
            "discrepancy_mode", "discrepancy definition", "\"binary\"");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "graph_discrepancy");
    }

    virtual shared_ptr<graph_discrepancy_search::GraphDiscrepancySearch>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            graph_discrepancy_search::GraphDiscrepancySearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<shared_ptr<Evaluator>>("prune_eval", nullptr),
            opts.get<bool>("reopen_closed"), opts.get<bool>("anytime"),
            opts.get<string>("discrepancy_mode"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<GraphDiscrepancySearchAliasFeature> _plugin_alias;
}
