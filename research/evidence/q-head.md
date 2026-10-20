# Corrected TF3: does the Q head help?

Subsequent user decision: daily-drive Q-on despite the inconclusive screen.
Ready package: `WORK/engines/vegamt-tf3-q/vegamt-tf3-q`.
This is the same tested binary/net with Q enabled by default; startup verified.
The measurements and conclusions below remain the experiment's original record.

Q-on scored **5.5–5.5** against Q-off: **1 wins, 9 draws, 1 losses** in 11 scored games at 10 seconds plus 0.1 seconds per move.

The user stopped the screen early. Eleven games completed; the twelfth was interrupted and is excluded from the score. Five opening pairs completed; one further completed game was a draw. This small screen shows no clear advantage for Q-on. It does not establish a small Elo gain, equivalence, or the best blending weight.

## What was changed

`NextActionValues` is the only treatment difference. Both use the original corrected Fixed 3 classic binary, two search threads, batch eight, two TensorRT lanes and the existing 4096-entry exact-input cache. Moves-left and outcome blending are off. No Root Scout or experimental cache candidate is involved. No original package or saved GUI option was changed.

Q supplies a guess for a move the search has not visited: 75% of the ordinary fallback estimate plus 25% of the action-head estimate. After a completed visit, the searched estimate takes over. This happens throughout the tree. The head does not dictate the final move, and it does not stay blended into searched values.

## Playing results

| Opening | Q-on as White | Q-on as Black | Q-on points / 2 |
|---|---|---|---:|
| p00 | draw | draw | 1 |
| p03 | draw | draw | 1 |
| p04 | draw | loss | 0.5 |
| p05 | win | draw | 1.5 |
| p08 | draw | draw | 1 |
| p09 | censored | draw | censored |

Opening pairs: 1 won, 3 tied, 1 lost by Q-on. Censored games: 1.

The planned twenty games used ten frozen openings with colors reversed; which treatment got White first alternated between pairs. Both engines had equal clocks and settings, complete legal opening histories, warm-up before the clocks, and no pondering. Only one engine was asked to search at a time. There was no evaluation-based resignation or adjudication.

Claimable draws ended automatically, including a draw that could be claimed by announcing the next repetition move. A 300-played-ply cap would have been recorded as censored, not silently counted as a draw.

Observed endings: threefold_repetition: 7, checkmate: 2, insufficient_material: 2, user_interrupted: 1. Prospective threefold claims: 7.

Independent replay checked **1,414 engine moves** (including the interrupted game's completed moves), every recorded clock budget, the PGN/result agreement, and **246,376 PV moves** across the full search updates. There were no engine crashes, illegal moves or time forfeits. The stop signal initially entered the harness's generic failure handler; the raw record is preserved and its administrative loss is explicitly reclassified as an unscored user interruption.

## Speed you would see in the GUI

These are native UCI NPS readings from ordinary two-thread/batch-eight searches. Two rounds used opposite order, fresh processes and three-second limits. The same positions were used in both treatments.

| Position | Q off NPS | Q on NPS | Paired change |
|---|---:|---:|---:|
| startpos | 6,247.0 | 6,273.5 | +0.42% |
| p03 | 6,300.0 | 6,238.0 | -0.99% |
| p05 | 6,294.5 | 6,194.0 | -1.60% |
| p15 | 5,976.5 | 6,042.5 | +1.11% |
| queen-sac | 4,763.5 | 5,091.5 | +6.90% |
| queen-sac-mirror | 4,963.0 | 4,920.5 | -0.85% |

Across the twelve paired measurements, the geometric mean Q-on/off ratio was **1.00794 (+0.79%)**. The table's NPS values are the median of the two readings per treatment. Different tree shapes can change NPS: this is playing-engine throughput, not an isolated GPU inference benchmark.

| Actual match searches of at least 50 ms | Median native NPS | Samples |
|---|---:|---:|
| Q off | 6,258 | 701 |
| Q on | 6,230 | 699 |

The game medians describe what the engine reported during play. They include different positions and time allocations, so the paired fixed-position measurements provide the cleaner speed comparison.

## Did the mechanism actually run?

Yes, on the tested positions. The 24 small full-search diagnostics checked 773 unvisited-edge scores and 123 visited-edge scores. All 448 Q-on root action estimates were finite and within WDL bounds. Q-off had no active action estimates. Maximum score-check residual: 0.00001007, below the unchanged 0.00004 tolerance.

Root policy priors remained the same. Root visit distributions differed in 5 of twelve position/budget comparisons. Final selected moves were the same in all twelve; the normal timed probes also selected the same move in both treatments.

Two concrete examples:

- From start position at 16 visits, Q-on examined b3 once; Q-off had not examined it. Both chose d4. At 128 visits their root visit counts had converged.
- In the queen-sacrifice fixture, both chose Bg2 at 16 visits and Qf4+ at 128. At 128, Q-on allocated 80 visits to Qf4+ versus 64 with Q-off, out of 127 child visits. Q is steering exploration, but this does not show an earlier tactical discovery. The color mirror also chose the correct queen move in both settings at 128.

Source review confirms the Fixed 3 traversal changes cover action-guided exploration outside policy order. This run samples actual full-search behavior; it does not prove the absence of every possible search bug.

## Test-reader corrections and reproducibility

The initial diagnostics exposed two mistaken assumptions in the checker, not changes in the engine. Visited Q is transformed for display using approximate WDL math, and an allocated child with zero visits prints a zero node value even though selection still uses FPU. AMENDMENT_001.md and AMENDMENT_002.md document the corrections. The checker now accounts for displayed precision and independently cross-checks the fallback against the parent. Artificially corrupting cold or visited selection values by 0.01 is rejected in both treatments.

All three searches completed before those interruptions were retained without rerunning them. The original failed results and logs remain in runs/main and runs/resumed. No game, opening, setting, tolerance or outcome rule was changed after results. The original 1200-second deadline included the interruptions.

Stopped at 2026-09-30T20:22:23.555151+00:00. Elapsed time from the first attempt: 925.0 seconds. Maximum sampled GPU temperature: 78 °C; maximum sampled GPU power: 60.07 W. Input hashes were verified unchanged after the experiment.

The daily-driver default remains Q-off. A larger match at a longer time control would be needed to justify changing that default; this screen alone is not sufficient to claim a dependable strength improvement.

## Files

- `plan.json`, `PROTOCOL.md`: frozen positions, settings, package/net/source hashes and decision scope.
- `runs/final/results.json`: all 48 actual probe results and all game outcomes, including the reused searches from earlier attempts.
- `runs/final/analysis.json`: independent audit, paired speed table and opening-pair scores.
- `runs/final/games.pgn`, `moves.jsonl`, `*.uci`, `*.stderr`, `gpu.csv`: complete games, move clocks, raw protocol and telemetry. Earlier-attempt logs are in the recorded `source_run` paths.
- `run.py`, `checks.py`, `analyze.py`: reproducible runner and readers.
- `tf3-q-on`, `tf3-q-off`: optional experiment launchers pointing at the same unchanged Fixed 3 package; the GUI can override their defaults through its saved UCI options.

Engine SHA-256: `3824cc57b9a7403bd13baabeb69799186271cea24e42c8f1957ae4ff38633482`.
Model SHA-256: `1534cbc469babebbb6225e0c5890ffeb984fc08e313fd84c78b674974d79e943`.
