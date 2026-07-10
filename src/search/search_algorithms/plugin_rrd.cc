#include "rrd_search.h"

#include "../plugins/plugin.h"
#include "../utils/logging.h"

using namespace std;

namespace plugin_rrd {
class RRDSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, rrd_search::RRDSearch> {
public:
    RRDSearchFeature() : TypedFeature("rrd") {
        document_title("Round-Robin on Distance");
        document_synopsis(
            "Round-Robin on distance (RR-D; Fickert, Gu & Ruml, AAAI 2022): a "
            "bounded-suboptimal best-first search that, like EES, separates the "
            "roles of its estimators, but rotates through its three queues in "
            "lockstep with the expansion counter instead of EES's priority "
            "cascade: distance-to-go dhat, inadmissible cost f^ = g + hhat, and "
            "an admissible f = g + h cleanup queue. Both focal queues hold the "
            "nodes with f <= w * f_min, so the first goal RR-D selects is "
            "provably within w times optimal for admissible h. The reference "
            "fast-downward-xes fork registers this as `alt_d`.");

        add_option<shared_ptr<Evaluator>>(
            "h",
            "admissible heuristic; f = g + h is the lower bound ordering the "
            "cleanup list and driving the suboptimality bound (e.g. lmcut())");
        add_option<shared_ptr<Evaluator>>(
            "hhat",
            "(potentially inadmissible) cost-to-go estimate; f^ = g + hhat "
            "orders the f^ focal queue (e.g. ff())");
        add_list_option<shared_ptr<Evaluator>>(
            "dhat",
            "optional (potentially inadmissible) distance-to-go estimate "
            "ordering the distance focal queue; give zero or one evaluator "
            "(defaults to hhat), e.g. dhat=[landmark_sum(lm_rhw(), "
            "transform=adapt_costs(one))]",
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
        add_search_algorithm_options_to_feature(*this, "rrd");
    }

    virtual shared_ptr<rrd_search::RRDSearch> create_component(
        const plugins::Options &opts) const override {
        vector<shared_ptr<Evaluator>> dhat_list =
            opts.get_list<shared_ptr<Evaluator>>("dhat");
        if (dhat_list.size() > 1) {
            cerr << "rrd: dhat takes at most one evaluator." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        shared_ptr<Evaluator> dhat =
            dhat_list.empty() ? nullptr : dhat_list.front();
        return plugins::make_shared_from_arg_tuples<rrd_search::RRDSearch>(
            opts.get<shared_ptr<Evaluator>>("h"),
            opts.get<shared_ptr<Evaluator>>("hhat"), dhat,
            opts.get<double>("w"), opts.get<bool>("debias"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<RRDSearchFeature> _plugin;
}
