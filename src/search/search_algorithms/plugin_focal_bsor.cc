#include "focal_bsor_search.h"

#include "../plugins/plugin.h"
#include "../utils/logging.h"

using namespace std;

namespace plugin_focal_bsor {
class FocalBSORSearchFeature
    : public plugins::TypedFeature<
          SearchAlgorithm, focal_bsor_search::FocalBSORSearch> {
public:
    FocalBSORSearchFeature() : TypedFeature("focal_bsor") {
        document_title("Focal bounded-suboptimal rectangle search");
        document_synopsis(
            "Focal-list variant of Bounded-Suboptimal Rectangle Search "
            "(Thomas et al., HSDIP 2026). Splits each depth level into a focal "
            "queue of nodes within the bound (f <= w * f_min_max), ordered by "
            "the distance-to-go d, and an f-ordered remainder; the beam only "
            "expands from focal, so it never spends the aspect-limited budget "
            "on nodes that cannot participate in a within-bound solution. Uses "
            "the running-max f_min as a monotone bound (unlike bsor, which "
            "uses the instantaneous f_min). Returns a solution provably within "
            "w times optimal (for admissible eval). Set rr=true for the "
            "round-robin variant.");

        add_option<shared_ptr<Evaluator>>(
            "eval",
            "cost estimate h; f = g + h orders the open list, the focal "
            "threshold and the suboptimality bound (e.g. ff())");
        add_list_option<shared_ptr<Evaluator>>(
            "dist",
            "optional distance-to-go estimate d that orders each level's "
            "focal queue; give zero or one evaluator (defaults to eval), "
            "e.g. dist=[lmcut()]",
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
            "round-robin variant: interleave a lowest-f expansion "
            "before each rectangle expansion",
            "false");
        add_option<bool>(
            "focal_expand_remainder",
            "when a level's focal queue is empty, expand its min-f "
            "remainder node rather than skipping the level",
            "false");
        add_search_pruning_options_to_feature(*this);
        add_search_algorithm_options_to_feature(*this, "focal_bsor");
    }

    virtual shared_ptr<focal_bsor_search::FocalBSORSearch> create_component(
        const plugins::Options &opts) const override {
        vector<shared_ptr<Evaluator>> dist_list =
            opts.get_list<shared_ptr<Evaluator>>("dist");
        if (dist_list.size() > 1) {
            cerr << "focal_bsor: dist takes at most one evaluator." << endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
        shared_ptr<Evaluator> dist =
            dist_list.empty() ? nullptr : dist_list.front();
        return plugins::make_shared_from_arg_tuples<
            focal_bsor_search::FocalBSORSearch>(
            opts.get<shared_ptr<Evaluator>>("eval"), dist,
            opts.get<double>("w"), opts.get<double>("aspect"),
            opts.get<bool>("rr"), opts.get<bool>("focal_expand_remainder"),
            get_search_pruning_arguments_from_options(opts),
            get_search_algorithm_arguments_from_options(opts));
    }
};

static plugins::FeaturePlugin<FocalBSORSearchFeature> _plugin;
}
