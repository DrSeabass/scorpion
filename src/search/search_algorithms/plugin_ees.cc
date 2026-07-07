#include "ees_search.h"

#include "../plugins/plugin.h"
#include "../utils/logging.h"

using namespace std;

namespace plugin_ees {
class EESSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, ees_search::EESSearch> {
public:
    EESSearchFeature() : TypedFeature("ees") {
        document_title("Explicit Estimation Search");
        document_synopsis(
            "Explicit Estimation Search (Thayer & Ruml, IJCAI 2011): a "
            "bounded-suboptimal best-first search that separates the roles of "
            "its estimators. It orders search by an inadmissible cost estimate "
            "f^ = g + hhat and a distance-to-go estimate dhat, while consulting "
            "an admissible h only to prove the w bound. Returns the first goal "
            "its selectNode rule chooses, which is provably within w times "
            "optimal for admissible h.");

        add_option<shared_ptr<Evaluator>>(
            "h",
            "admissible heuristic; f = g + h is the lower bound ordering the "
            "cleanup list and driving the suboptimality bound (e.g. lmcut())");
        add_option<shared_ptr<Evaluator>>(
            "hhat",
            "(potentially inadmissible) cost-to-go estimate; f^ = g + hhat "
            "orders the open list (e.g. ff())");
        add_list_option<shared_ptr<Evaluator>>(
            "dhat",
            "optional (potentially inadmissible) distance-to-go estimate "
            "ordering the focal list; give zero or one evaluator (defaults to "
            "hhat), e.g. dhat=[landmark_sum(lm_rhw(), transform=adapt_costs("
            "one))]",
            "[]");
        add_option<double>(
            "w",
            "suboptimality bound (>= 1); the returned solution costs at most w "
            "times the optimum when h is admissible",
            "1.0");
        add_option<bool>(
            "debias",
            "correct hhat and dhat for the mean single-step error along each "
            "node's root-to-node path (Thayer, Dionne & Ruml 2011), turning the "
            "base estimates into sharper inadmissible ones; off leaves them "
            "verbatim",
            "false");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "ees");
    }

    virtual shared_ptr<ees_search::EESSearch> create_component(
        const plugins::Options &opts) const override {
        vector<shared_ptr<Evaluator>> dhat_list =
            opts.get_list<shared_ptr<Evaluator>>("dhat");
        if (dhat_list.size() > 1) {
            cerr << "ees: dhat takes at most one evaluator." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        shared_ptr<Evaluator> dhat =
            dhat_list.empty() ? nullptr : dhat_list.front();
        return plugins::make_shared_from_arg_tuples<ees_search::EESSearch>(
            opts.get<shared_ptr<Evaluator>>("h"),
            opts.get<shared_ptr<Evaluator>>("hhat"), dhat,
            opts.get<double>("w"), opts.get<bool>("debias"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<EESSearchFeature> _plugin;
}
