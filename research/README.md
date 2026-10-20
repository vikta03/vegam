# Experiments, including the ideas that did not win

The daily driver is the corrected classic engine. Experimental outcomes below
are retained separately from its defaults.

| Question | Result | What it supports |
|---|---|---|
| Did corrected classic search run at useful laptop speed? | Saved five-position readings around 5,153–6,538 NPS | A small hardware-specific throughput snapshot |
| Was DAG stronger in the screen? | 4 wins, 12 draws, 4 losses across 20 games | No observed advantage in that short screen |
| Did Q-on establish a strength gain? | 1 win, 9 draws, 1 loss in the stopped 11-game screen | No established gain; Q remains experimental |
| Can internals be inspected against the actual engine? | 64-position calibration and saved traces | Bounded numerical calibration, not universal equivalence |

## Fixed3 throughput

RTX 3050 Laptop GPU, 4 GiB VRAM; two search threads, minibatch eight,
512 MiB search limit; TensorRT, two inference lanes. Two runs per position,
mode order reversed in the second round. Values are median final UCI `nps`.

| Position | Classic | DAG |
|---|---:|---:|
| Start position | 6,320.5 | 7,430 |
| Retained e4/e5 position | 6,337 | 7,034 |
| Middlegame p03 | 6,218 | 6,670 |
| Endgame p15 | 6,538 | 8,047 |
| Queen-sacrifice position | 5,153 | 5,580 |

Smart pruning remained enabled. Requested five-second searches could stop
earlier; the start-position runs lasted about 4.33–4.36 seconds. The retained
e4/e5 case used real clock inputs. DAG visits can reuse work through
transpositions, so a larger visit count is not the same as faster fresh inference.

The 20-game comparison used 5+0.05, ten openings with colors swapped, and no
evaluation adjudication. All 2,145 engine moves were replayed. The saved results
reported no illegal moves, crashes or time forfeits.

The Q comparison is a different, later experiment. Its aggregate result must
not be conflated with the DAG comparison or treated as proof of equivalence.

## Included evidence

- [Position-by-position throughput and PVs](evidence/fixed3-performance.json)
- [Full-search correction checks](evidence/fixed3-search.json)
- [20-game results](evidence/fixed3-games.json), [PGN](evidence/fixed3-games.pgn)
  and [independent replay](evidence/fixed3-replay.json)
- [Q-head report](evidence/q-head.md)
- [64-position calibration](evidence/calibration.json)
- [Illustrative Catalan game](evidence/catalan.pgn)
- [Book measurements and causal interventions](../book/data/measurements.json)

Workstation paths in these publication copies are replaced by logical source
identifiers. `provenance/import-manifest.json` records the selected original
files and hashes. No private conversation transcript or raw training corpus
is included.
