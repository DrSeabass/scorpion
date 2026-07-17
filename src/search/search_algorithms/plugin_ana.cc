#include "anytime_nonparametric_search.h"

#include "../plugins/plugin.h"

using namespace std;

namespace plugin_ana {
class AnytimeNonparametricSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm,
                                   anytime_nonparametric_search::AnytimeNonparametricSearch> {
public:
    AnytimeNonparametricSearchFeature() : TypedFeature("ana") {
        document_title("Anytime nonparametric A* (ANA*)");
        document_synopsis(
            "Eager best-first search that expands the open node with maximal "
            "potential e(s) = (G - g(s)) / h(s), where G is the cost of the "
            "current incumbent (initially infinity). It needs no weight "
            "parameter: with G = infinity the first-solution search is "
            "maximally greedy (h-only), and each new incumbent tightens G so "
            "the search automatically re-greedifies toward improving it. Nodes "
            "with g(s) + h(s) >= G are pruned. See van den Berg, Shah, Huang & "
            "Goldberg, Anytime Nonparametric A*, AAAI 2011. Stops after the "
            "first plan is found unless anytime=true.");

        add_option<shared_ptr<Evaluator>>("eval", "ranking evaluator (h)");
        add_option<bool>(
            "reopen_closed",
            "reopen closed nodes if a cheaper path is found (ANA* relies on "
            "this to propagate improved g-values)",
            "true");
        add_option<bool>(
            "anytime",
            "continue search after finding a solution to improve the incumbent",
            "false");
        add_option<bool>(
            "prune_with_h",
            "prune nodes that cannot improve the incumbent using f = g + h "
            "(g(s) + h(s) >= G). This is only sound when eval is admissible; "
            "set to false to prune on g alone when eval may overestimate.",
            "true");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "ana");
    }

    virtual shared_ptr<anytime_nonparametric_search::AnytimeNonparametricSearch>
    create_component(const plugins::Options &opts) const override {
        return plugins::make_shared_from_arg_tuples<
            anytime_nonparametric_search::AnytimeNonparametricSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"),
            opts.get<bool>("reopen_closed"),
            opts.get<bool>("anytime"),
            opts.get<bool>("prune_with_h"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<AnytimeNonparametricSearchFeature> _plugin;
}
