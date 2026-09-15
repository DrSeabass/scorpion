#ifndef SEARCH_ALGORITHMS_TRIANGLE_SWEEP_START_H
#define SEARCH_ALGORITHMS_TRIANGLE_SWEEP_START_H

#include "../utils/rng.h"
#include "../utils/rng_options.h"
#include "../utils/system.h"

#include <algorithm>
#include <memory>
#include <iostream>

namespace triangle_sweep_start {

// Sweep starts are absolute depths, even when empty prefix layers were removed.
class SweepStart {
    bool random_start;
    int window;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

public:
    explicit SweepStart(bool random_start = false, int random_seed = -1, int window = -1)
        : random_start(random_start), window(window),
          rng(random_start ? utils::get_rng(random_seed) : nullptr) {
        if (window < -1 || (random_start && window >= 0)) {
            std::cerr << "window must be -1 (disabled) or nonnegative; "
                         "random_start and window cannot be combined." << std::endl;
            utils::exit_with(utils::ExitCode::SEARCH_INPUT_ERROR);
        }
    }

    bool enabled() const { return random_start || window >= 0; }

    int choose(int depth_offset, int deepest_layer) {
        if (window >= 0)
            return std::max(0, deepest_layer - window);
        if (!random_start)
            return 0;
        int absolute_start = rng->random(depth_offset + deepest_layer + 1);
        return std::max(0, absolute_start - depth_offset);
    }
};

}

#endif
