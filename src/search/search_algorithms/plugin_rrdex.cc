#include "rrdex_search.h"

#include "../plugins/plugin.h"
#include "../utils/logging.h"

using namespace std;

namespace plugin_rrdex {
class RRDEXSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, rrdex_search::RRDEXSearch> {
public:
    RRDEXSearchFeature() : TypedFeature("rrdex") {
        document_title("Round-Robin Dynamic Expected Effort Search");
        document_synopsis(
            "Round-robin dynamic expected effort search (RR-DXES; Fickert, Gu & "
            "Ruml, AAAI 2022): the reference fast-downward-xes `dxes` run with "
            "alternation_mode=BOTH and default variance settings. Like RR-D it "
            "rotates through three queues (distance, f^ = g + hhat, and an "
            "admissible f = g + h cleanup queue) in lockstep with the expansion "
            "counter, and both focal queues hold the nodes with f <= w * f_min "
            "so the first goal selected is within w times optimal for admissible "
            "h. Unlike RR-D, the first queue is ordered by dynamic expected "
            "effort dhat / P(within bound) rather than raw distance, where P is "
            "DXES's Gaussian belief that a node yields a within-bound solution.");

        add_option<shared_ptr<Evaluator>>(
            "h",
            "admissible heuristic; f = g + h is the lower bound ordering the "
            "cleanup list and driving the suboptimality bound (e.g. lmcut())");
        add_option<shared_ptr<Evaluator>>(
            "hhat",
            "(potentially inadmissible) cost-to-go estimate; f^ = g + hhat "
            "orders the f^ focal queue and feeds the expected-effort belief "
            "(e.g. ff())");
        add_list_option<shared_ptr<Evaluator>>(
            "dhat",
            "optional (potentially inadmissible) distance-to-go estimate; the "
            "numerator of expected effort; give zero or one evaluator (defaults "
            "to hhat), e.g. dhat=[landmark_sum(lm_rhw(), "
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
        add_search_algorithm_options_to_feature(*this, "rrdex");
    }

    virtual shared_ptr<rrdex_search::RRDEXSearch> create_component(
        const plugins::Options &opts) const override {
        vector<shared_ptr<Evaluator>> dhat_list =
            opts.get_list<shared_ptr<Evaluator>>("dhat");
        if (dhat_list.size() > 1) {
            cerr << "rrdex: dhat takes at most one evaluator." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        shared_ptr<Evaluator> dhat =
            dhat_list.empty() ? nullptr : dhat_list.front();
        return plugins::make_shared_from_arg_tuples<rrdex_search::RRDEXSearch>(
            opts.get<shared_ptr<Evaluator>>("h"),
            opts.get<shared_ptr<Evaluator>>("hhat"), dhat,
            opts.get<double>("w"), opts.get<bool>("debias"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<RRDEXSearchFeature> _plugin;
}
