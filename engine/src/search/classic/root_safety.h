// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "chess/position.h"

namespace lczero::classic {

// Restrict only rules-proven losses. Match the main search's terminal ordering:
// checkmate first, then draws, before considering an opponent reply.
inline MoveList ImmediateMateSafety(const PositionHistory& history,
                                    MoveList candidates) {
  if (candidates.empty()) {
    candidates = history.Last().GetBoard().GenerateLegalMoves();
  }
  PositionHistory next_history(history);
  const auto root_length = history.GetLength();
  next_history.Reserve(root_length + 1);
  MoveList safe, wins;
  for (auto move : candidates) {
    next_history.Trim(root_length);
    next_history.Append(move);
    const auto& position = next_history.Last();
    const auto& child = position.GetBoard();
    const auto replies = child.GenerateLegalMoves();
    if (replies.empty() && child.IsUnderCheck()) {
      wins.push_back(move);
      continue;
    }
    if (replies.empty() || !child.HasMatingMaterial() ||
        position.GetRule50Ply() >= 100 || position.GetRepetitions() >= 2) {
      safe.push_back(move);
      continue;
    }
    bool loses = false;
    for (auto reply : replies) {
      ChessBoard next(child);
      next.ApplyMove(reply);
      next.Mirror();
      if (next.IsUnderCheck() && next.GenerateLegalMoves().empty()) {
        loses = true;
        break;
      }
    }
    if (!loses) safe.push_back(move);
  }
  if (!wins.empty()) return wins;
  if (!safe.empty()) return safe;
  return candidates;
}

}  // namespace lczero::classic
