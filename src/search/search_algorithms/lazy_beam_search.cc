#include "lazy_beam_search.h"

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

namespace lazy_beam_search {

LazyBeamSearch::LazyBeamSearch(
    const shared_ptr<Evaluator> &eval,
    int beam_width,
    const shared_ptr<PruningMethod> &pruning,
    OperatorCost cost_type, int bound, double max_time,
    const string &description, utils::Verbosity verbosity)
    : SearchAlgorithm(cost_type, bound, max_time, description, verbosity),
      beam_width(beam_width),
      eval(eval),
      pruning_method(pruning) {
    if (beam_width <= 0) {
        cerr << "LazyBeamSearch: beam_width must be positive." << endl;
        utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
    }
}

void LazyBeamSearch::initialize() {
    log << "Conducting lazy beam search with beam width " << beam_width
        << ", (real) bound = " << bound << endl;

    assert(eval);

    set<Evaluator *> evals;
    eval->get_path_dependent_evaluators(evals);
    path_dependent_evaluators.assign(evals.begin(), evals.end());

    State initial_state = state_registry.get_initial_state();
    for (Evaluator *evaluator : path_dependent_evaluators) {
        evaluator->notify_initial_state(initial_state);
    }

    beam.clear();
    beam.push_back(initial_state.get_id());

    pruning_method->initialize(task);
}

void LazyBeamSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
}

SearchStatus LazyBeamSearch::step() {
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

    struct Candidate {
        StateID id;
        int h;
        int g;
    };
    vector<Candidate> candidates;

    for (StateID id : beam) {
        State s = state_registry.lookup_state(id);
        SearchNode node = search_space.get_node(s);

        if (node.is_closed())
            continue;

        node.close();
        statistics.inc_expanded();

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(s, applicable_ops);
        pruning_method->prune_operators(s, applicable_ops);

        for (OperatorID op_id : applicable_ops) {
            OperatorProxy op = task_proxy.get_operators()[op_id];
            // Scorpion SearchNode lacks get_real_g(); get_g() equals real_g for NORMAL cost type.
            if (node.get_g() + op.get_cost() >= bound)
                continue;

            State succ_state = state_registry.get_successor_state(s, op);
            statistics.inc_generated();

            SearchNode succ_node = search_space.get_node(succ_state);

            for (Evaluator *evaluator : path_dependent_evaluators) {
                evaluator->notify_state_transition(s, op_id, succ_state);
            }

            if (succ_node.is_dead_end() || succ_node.is_closed())
                continue;

            int succ_g = node.get_g() + get_adjusted_cost(op);

            if (succ_node.is_new()) {
                EvaluationContext succ_eval_context(succ_state, succ_g, false, &statistics);
                statistics.inc_evaluated_states();
                int h = succ_eval_context.get_evaluator_value_or_infinity(eval.get());
                if (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
                    succ_node.mark_as_dead_end();
                    statistics.inc_dead_ends();
                    continue;
                }
                succ_node.open_new_node(node, op, get_adjusted_cost(op));
                candidates.push_back({succ_state.get_id(), h, succ_g});
            } else {
                if (succ_node.get_g() > succ_g) {
                    succ_node.update_open_node_parent(node, op, get_adjusted_cost(op));
                }
                EvaluationContext succ_eval_context(succ_state, succ_node.get_g(), false, &statistics);
                statistics.inc_evaluated_states();
                int h = succ_eval_context.get_evaluator_value_or_infinity(eval.get());
                if (h == EvaluationResult::INFTY && eval->dead_ends_are_reliable()) {
                    succ_node.mark_as_dead_end();
                    statistics.inc_dead_ends();
                    continue;
                }
                candidates.push_back({succ_state.get_id(), h, succ_node.get_g()});
            }

            if (check_goal_and_set_plan(succ_state)) {
                return SOLVED;
            }
        }
    }

    if (candidates.empty()) {
        log << "All successors pruned by beam search -- no solution!" << endl;
        return FAILED;
    }

    sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
        if (a.h != b.h)
            return a.h < b.h;
        return a.g < b.g;
    });

    vector<StateID> next_beam;
    next_beam.reserve(min<int>(beam_width, candidates.size()));

    for (const Candidate &cand : candidates) {
        if (static_cast<int>(next_beam.size()) == beam_width)
            break;

        State succ_state = state_registry.lookup_state(cand.id);
        SearchNode succ_node = search_space.get_node(succ_state);

        if (succ_node.is_dead_end() || succ_node.is_closed())
            continue;

        bool already_in_beam = false;
        for (StateID id_in_beam : next_beam) {
            if (id_in_beam == cand.id) {
                already_in_beam = true;
                break;
            }
        }
        if (already_in_beam)
            continue;

        next_beam.push_back(cand.id);
    }

    beam = move(next_beam);
    return IN_PROGRESS;
}

}
