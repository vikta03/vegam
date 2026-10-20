// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>

namespace lczero::next_heads {
// Diagnostic only: whether a nonzero duration utility was evaluated in play.
inline std::atomic<bool> moves_left_activated{false};

inline float MedianPlies(float mu) {
  return std::expm1(std::clamp(mu, 0.0f, std::log1p(512.0f)));
}

// The existing Lc0 utility equations, shared with the synthetic regression.
// Caller retains the parent |Q| threshold and terminal-node handling.
inline float MovesLeftUtility(float child_m, float parent_m, float q,
                              float slope, float cap, float threshold,
                              float constant, float linear, float square) {
  float m = std::clamp(slope * (child_m - parent_m), -cap, cap);
  m *= std::signbit(-q) ? -1.0f : 1.0f;
  if (threshold > 0.0f && threshold < 1.0f) {
    q = std::max(0.0f, std::abs(q) - threshold) / (1.0f - threshold);
  }
  m *= constant + linear * std::abs(q) + square * q * q;
  return m;
}
}  // namespace lczero::next_heads
