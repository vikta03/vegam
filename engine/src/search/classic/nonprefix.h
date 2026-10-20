// GPL-3.0-or-later with the Lc0 NVIDIA exception.
#pragma once
#include <algorithm>
#include "search/classic/node.h"

namespace lczero::classic {
inline unsigned CountExcludedRootMoves(const Node& node, const MoveList& allowed) {
  if (allowed.empty()) return 0;
  unsigned excluded = 0;
  for (const auto& edge : node.Edges())
    if (std::find(allowed.begin(), allowed.end(), edge.GetMove()) == allowed.end())
      ++excluded;
  return excluded;
}

// Call only with the exclusive tree lock. A fresh node's packed flags may
// still be read by its lock-free evaluator, so never write before first backup.
inline void PreserveNonprefixChildren(Node& node, bool action_values,
                                      bool excludes_legal_moves) {
  if (node.GetN() > 0 && !node.HasNonprefixChildren() &&
      (action_values || excludes_legal_moves)) node.MarkNonprefixChildren();
}
}  // namespace lczero::classic
