#include "bsor_search.h"

#include "../plugins/plugin.h"
#include "../utils/logging.h"

using namespace std;

namespace plugin_bsor {
class BSORSearchFeature
    : public plugins::TypedFeature<SearchAlgorithm, bsor_search::BSORSearch> {
public:
    BSORSearchFeature() : TypedFeature("bsor") {
        document_title("Bounded-suboptimal rectangle search");
        document_synopsis(
            "Bounded-Suboptimal Rectangle Search (Thomas et al., HSDIP 2026): "
            "a bounded-suboptimal adaptation of Rectangle Search. Returns a "
            "solution provably within w times optimal (for admissible eval). "
            "Set rr=true for the Round-Robin Rectangle (RRR) variant, which "
            "interleaves lowest-f expansions to raise f_min faster.");

        add_option<shared_ptr<Evaluator>>(
            "eval",
            "cost estimate h; f = g + h orders the open list and drives "
            "the suboptimality bound (e.g. ff())");
        add_list_option<shared_ptr<Evaluator>>(
            "dist",
            "optional distance-to-go estimate d that orders each "
            "rectangle depth bucket; give zero or one evaluator (defaults to "
            "eval), e.g. dist=[lmcut()]",
            "[]");
        add_option<double>(
            "w",
            "suboptimality bound (>= 1); the returned solution costs at "
            "most w times the optimum when eval is admissible",
            "1.0");
        add_option<double>(
            "aspect",
            "rectangle aspect ratio a (> 0): a >= 1 explores deeper, "
            "a < 1 explores wider",
            "1.0");
        add_option<bool>(
            "rr",
            "round-robin variant (RRR): interleave a lowest-f expansion "
            "before each rectangle expansion",
            "false");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "bsor");
    }

    virtual shared_ptr<bsor_search::BSORSearch> create_component(
        const plugins::Options &opts) const override {
        vector<shared_ptr<Evaluator>> dist_list =
            opts.get_list<shared_ptr<Evaluator>>("dist");
        if (dist_list.size() > 1) {
            cerr << "bsor: dist takes at most one evaluator." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        shared_ptr<Evaluator> dist =
            dist_list.empty() ? nullptr : dist_list.front();
        return plugins::make_shared_from_arg_tuples<bsor_search::BSORSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"), dist,
            opts.get<double>("w"), opts.get<double>("aspect"),
            opts.get<bool>("rr"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<BSORSearchFeature> _plugin;
}
