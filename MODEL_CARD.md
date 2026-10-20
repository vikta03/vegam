# VEGAM wide epoch 3

**Author:** Remmy Victor (@vikta03)  
**Original weights licence:** CC BY 4.0  
**Intended use:** UCI chess play and investigation of a small learned evaluator.

| Property | Value |
|---|---|
| Parameters | 9,800,012 |
| Blocks / width / heads / FFN | 8 / 384 / 12 / 576 |
| Adaptive relation rank | 16 |
| Input | 64 × 120; eight board-history frames |
| Candidate capacity | 256 legal-move slots |
| Outputs | Policy; state WDL; action WDL; outcome; duration |
| Training initialization | Random model → 5 million-position pilot → three full training passes |
| Main-pass exposures | 732,938,670 |
| Lineage exposures including pilot | 737,938,670 |
| Rating | Unrated; no established external Elo |

The indexed corpus contained 402 archives, 2,108,260 games and 249,309,004
stored records: 244,312,890 training, 2,509,537 validation and 2,486,577 test.
Exposure counts are not unique positions, and records are not necessarily
unique chess boards.

The teacher policy and best-Q/draw targets come from Lc0 V6 records.
Action-WDL supervision is sparse. Duration targets describe recorded
remaining play, not a human annotated strategic plan.

The raw corpus and full optimizer checkpoints are not distributed here.
Weights-only EMA, fallback ONNX and mixed-precision TensorRT-source assets
are distributed through the release, each with a checksum.

## Known limits

- High predicted win probability is not a proof of a forced win or successful
  conversion. Evaluation calibration, history sensitivity and difficult
  endgames remain research topics.
- The Q head changes exploration. The small retained screen did not establish
  a strength advantage.
- TensorRT arithmetic differs from the fallback implementation. The shipped
  mixed-precision design was selected after numerical checks; a new GPU/runtime
  still needs local validation.
- The history-free FEN representation can differ from the same board reached
  through actual moves.
- Attention visualizations show routing weights. They alone do not prove the
  presence of a named chess concept or explain causality.

See the [book](book/README.md), [training recipe](docs/TRAINING.md) and
[experiment evidence](research/README.md) for concrete examples and test boundaries.
