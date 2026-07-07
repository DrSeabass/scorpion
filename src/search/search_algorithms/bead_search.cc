#include "bead_search.h"

#include "../evaluation_context.h"
#include "../evaluation_result.h"
#include "../evaluator.h"
#include "../pruning_method.h"

#include "../task_utils/successor_generator.h"
#include "../task_utils/task_properties.h"
#include "../utils/logging.h"

#include <algorithm>
#include <cassert>

using namespace std;

namespace bead_search {

BeadSearch::BeadSearch(
    const shared_ptr<Evaluator> &eval, int beam_width,
    const shared_ptr<PruningMethod> &pruning, OperatorCost cost_type, int bound,
    double max_time, const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      beam_width(beam_width),
      eval(eval),
      pruning_method(pruning) {
    if (beam_width <= 0) {
        cerr << "BeadSearch: beam_width must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void BeadSearch::initialize() {
    log << "Conducting eager Bead search with beam width " << beam_width
        << ", (real) bound = " << bound << endl;

    assert(eval);

    set<Evaluator *> evals;
    eval->get_path_dependent_evaluators(evals);
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    bool is_dead_end = false;
    int h = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
        is_dead_end = true;
    }

    if (is_dead_end) {
        log << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context)) {
            statistics.print_checkpoint_line(0);
        }
        start_evaluator_statistics(eval_context);

        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();

        beam.clear();
        beam.push_back(initial_state.get_id());
    }

    print_initial_evaluator_values(eval_context);
    pruning_method->initialize(task);
}

void BeadSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

namespace {
struct Candidate {
    StateID id;
    // Ranking value from `eval`; Bead treats it as a distance-to-go (d^)
    // estimate.
    int eval_value;
    int g;
};
}

SearchStatus BeadSearch::step() {
    if (beam.empty()) {
        log << "Beam is empty -- no solution!" << endl;
        return FAILED;
    }

    for (StateID id : beam) {
        State s = state_registry.lookup_state(id);
        if (check_goal_and_set_plan(s)) {
            return SOLVED;
        }
    }

    vector<Candidate> candidates;

    for (StateID id : beam) {
        State s = state_registry.lookup_state(id);
        SearchNode node = search_space.get_node(s);

        if (node.is_closed()) {
            continue;
        }

        node.close();
        statistics.inc_expanded();

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(s, applicable_ops);
        pruning_method->prune_operators(s, applicable_ops);

        for (OperatorID op_id : applicable_ops) {
            OperatorProxy op = task_proxy.get_operators()[op_id];

            // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for
            // NORMAL cost type.
            if (node.get_g() + op.get_cost() >= bound)
                continue;

            State succ_state = state_registry.get_successor_state(s, op);
            statistics.inc_generated();

            SearchNode succ_node = search_space.get_node(succ_state);

            for (Evaluator *evaluator : path_dependent_evaluators) {
                evaluator->notify_state_transition(s, op_id, succ_state);
            }

            if (succ_node.is_dead_end())
                continue;

            // Bead drops duplicates: any state we have already generated (open
            // or closed) is skipped, even if reached by a cheaper path. This is
            // what keeps Bead unboundedly suboptimal; we neither reopen closed
            // nodes nor update the parent of open ones.
            if (!succ_node.is_new())
                continue;

            int succ_g = node.get_g() + get_adjusted_cost(op);

            EvaluationContext succ_eval_context(
                succ_state, succ_g, false, &statistics);
            statistics.inc_evaluated_states();

            int eval_value =
                succ_eval_context.get_evaluator_value_or_infinity(eval.get());
            if (eval_value == EvaluationResult::INFTY &&
                eval->dead_ends_are_reliable()) {
                succ_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                continue;
            }

            succ_node.open_new_node(node, op, get_adjusted_cost(op));

            if (search_progress.check_progress(succ_eval_context)) {
                statistics.print_checkpoint_line(succ_node.get_g());
            }

            candidates.push_back({succ_state.get_id(), eval_value, succ_g});

            if (check_goal_and_set_plan(succ_state)) {
                return SOLVED;
            }
        }
    }

    if (candidates.empty()) {
        log << "All successors pruned by Bead search -- no solution!" << endl;
        return FAILED;
    }

    // Rank by the distance-to-go estimate (ascending), breaking ties on g so
    // that, among equally promising states, cheaper plans are preferred.
    sort(
        candidates.begin(), candidates.end(),
        [](const Candidate &a, const Candidate &b) {
            if (a.eval_value != b.eval_value)
                return a.eval_value < b.eval_value;
            return a.g < b.g;
        });

    vector<StateID> next_beam;
    next_beam.reserve(min<int>(beam_width, candidates.size()));

    for (const Candidate &cand : candidates) {
        if (static_cast<int>(next_beam.size()) == beam_width)
            break;
        // Duplicates were dropped at generation time, so every candidate is a
        // distinct, freshly opened state; no in-beam dedup is needed.
        next_beam.push_back(cand.id);
    }

    beam.swap(next_beam);
    return IN_PROGRESS;
}

void BeadSearch::start_evaluator_statistics(EvaluationContext &eval_context) {
    int value = eval_context.get_evaluator_value_or_infinity(eval.get());
    if (value != EvaluationResult::INFTY) {
        statistics.report_f_value_progress(value);
    }
}

}
