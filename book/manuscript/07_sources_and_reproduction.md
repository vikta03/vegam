# Appendix E. Source map and reproducibility
@objectives Locate the implementation behind important claims; identify every measurement family; reproduce the arithmetic and figures without starting a hardware experiment; know what a fresh engine reproduction requires.

## E.1 Source roots and the precedence rule

This edition was checked on 9 October 2026. The trained checkpoint configuration and current launcher take precedence over historical prose. Frozen source tied to the delivered binary takes precedence over a newer generic upstream description of search. Measurement receipts are interpreted with their own device, precision and settings rather than silently merged.

The source index retains three historical roots. **NEXT** maps to this repository's root for model and training source. **FIXED** maps to `engine/`. **WORK** identifies the original research archive; selected public receipts are in `research/evidence/` and `book/inputs/`. The companion `data/source-index.json` preserves historical identifiers, line counts and original hashes; `data/publication-source-index.json` maps the included source files.

## E.2 Implementation references

**S1 - The actual neural equations.** `NEXT/nextchess/model.py`. Config, lines 9-18, defines default values; the trusted checkpoint supplies width 384, heads 12 and FFN 576. Block.__init__, lines 21-35, defines normalization, projections, MLP and adaptive factors. Block.forward, lines 37-50, defines QKV reshape, bias construction, attention and residual updates. Net.__init__, lines 53-63, defines the input and output modules. Net.forward, lines 69-78, defines global pooling, move gathering and joint decoding. losses, lines 81-97, defines the five prediction objectives.

**S2 - Native input and output adapter.** `FIXED/src/neural/next_backend.cc`. Encode, lines 42-82, constructs the 64 x 120 features, padded move tuples and mask and supports input dumping. Trace, lines 85-93, records native outputs. Wdl, lines 103-107, performs stable three-way normalization. The later backend evaluation code converts normalized head results into search q, d, policy and optional action estimates. This is the frozen adapter used in the delivered source lineage.

**S3 - History encoding.** `FIXED/src/neural/encoder.cc`, EncodePositionForNN, especially lines 235-305. It walks the history, handles orientation, writes six own and six opposing piece planes plus repetition, and handles the limited en-passant reconstruction case for absent prior history.

**S4 - Training-record interpretation.** `NEXT/nextchess/raw.py`, wdl and decode, lines 70-151, defines target conversion, legality collection, castling identities, sparse action labels and duration/result masks. `NEXT/nextchess/fast_raw.py` provides the accelerated equivalent used with cloud context; `NEXT/nextchess/cloud_data.py` supplies retained game-prefix and frame context. Use the fast path's source alongside the readable decoder when auditing the actual training pipeline.

**S5 - Optimization and training state.** `NEXT/nextchess/cloud_train.py`, objective_sums and update, lines 55-130, defines masked objectives, distributed accounting, mixed-precision retry and gradient clipping. update_ema, lines 132-134, and the training loop around lines 298-308 implement the schedule and moving average. Construction around lines 175-193 sets the wide configuration and AdamW. Checkpoint save/resume retains optimizer, model, EMA, reader and counters.

**S6 - Daily execution identity.** `WORK/engines/vegamt-tf3-q/vegamt-tf3-q`, lines 1-9; `lc0.config`, lines 1-13; and `manifest.json`. The wrapper explicitly invokes classic search and enables action values while disabling moves-left and outcome blending. It loads the model beside the engine through resolved paths. The model origin is the wide-epoch3-trt3-io package, while the binary is the corrected Fixed 3 lineage.

**S7 - Search, perspective and score display.** `FIXED/src/search/classic/search.cc`. WDLRescale, lines 224-257; UCI reporting and score branches, lines 275-343; GetFpu and ComputeCpuct, lines 429-452; selection around lines 1690-1730. Read the whole surrounding functions for terminal handling, filters, in-flight visits and selection shortcuts; the book's PUCT equation is deliberately a simplified view of the mechanism.

**S8 - Cold-edge action advice.** `FIXED/src/search/classic/node.h`, Edge::ColdQ, lines 99-104; EdgeAndNode::GetQ, lines 390-391; GetU, lines 421-425. These locations establish the 75/25 fallback blend, takeover by completed searched values and the exploration denominator using started visits.

**S9 - Instrumentation.** `WORK/research/vegam_atlas/atlas/model.py`, Diagnostic.forward, beginning at line 69, reproduces the source equations with named taps and reversible interventions. `atlas/analysis.py`, `native.py`, `chessdata.py` and `storage.py` define saved inputs, native interfaces, game reconstruction and identity checks. The instrumentation is a diagnostic path, not a replacement playing engine.

## E.3 Training and engine receipts

**T1 - Completed training.** `NEXT/runs/epochs23-20260919-pro5000-continue/downloads/final/model.json`, `result.json` and `latest.pt`, plus `completion-verified.json` in the parent run directory. These identify the three-pass EMA, 732,938,670 main exposures, 715,761 steps and five-million-exposure pilot lineage. The final cloud ONNX hash differs from the later audited graph's hash; the book does not equate all exports byte for byte.

**T2 - Paired training-stage comparison.** `NEXT/runs/epoch-comparison-20260920/comparison.md`, `heldout.json` and per-game statistics. The same 32,768 validation records from 308 games support the epoch-1 versus epoch-3 prediction comparison. It is not a strength match.

**T3 - Corrected Q screen.** `WORK/research/tf3_qhead_20261001/RESULTS.md`. Eleven scored games, mechanism checks and normal-search NPS observations. The stopped twelfth game is excluded. This source does not establish that Q-on is superior or cannot hurt.

**T4 - Fixed 3 and optional DAG.** `WORK/research/vegamt_fixed3/LATEST_RESULTS.md` and the run's `RESULTS.md`, source receipt and full-search/performance/match subdirectories. These document 32 bounded CPU full-search checks, the two-mode speed screen and twenty-game result. Classic remains the selected daily launcher mode.

**Historical comparison.** `NEXT/VEGAM8.md` is the preserved earlier one-pass narrative. `NEXT/ENGINE-GUIDE.md` provides a later source-checked guide, but this textbook still checks the current launcher, checkpoint and frozen search source rather than treating any prose document as the final executable specification.

## E.4 Measurement receipts used in this book

**G1 - The chess game.** `WORK/research/vegam_atlas/fixtures/catalan.pgn`. The complete game has 201 legal plies and ends with 101.Qh2#. The anchor is reached after 160 plies, immediately before White's 81st move. Its position identifier is `b17cbc321220a5b0cd7c1480`.

**M1 - The 64-position calibration.** `WORK/research/vegam_atlas/data/runs/20261009T163532Z-354d7be2/`, with the current calibration record in `data/calibration.json`. All 64 positions passed the fixed 0.01 normalized-probability tolerance. Instrumented and original PyTorch raw outputs agreed exactly in the check. Maximum normalized discrepancies versus native were 0.006070632 policy, 0.005173182 state WDL and 0.006197649 action WDL. All 64 leading moves agreed. This is bounded numerical validation, not universal bitwise parity or a game-strength result.

**M2 - The anchor's detailed CPU study.** `WORK/research/vegam_atlas/data/runs/20261009T172318Z-bg7-6c1903/`. The important files are `result.json`, `trace.npz`, `interventions.json`, `request.json` and `status.json`. The run took about 4.27 seconds with a recorded peak process-group memory of about 651 MiB. It used float32 CPU arithmetic with float16-rounded EMA weights and reused the earlier native reference. The attempted GPU run stopped before evaluation because another application was using the GPU; it is not counted as a new GPU measurement.

For M2, maximum normalized discrepancies versus the saved native reference were approximately 0.000405127 policy, 0.000320468 state WDL and 0.001475962 action WDL. These correspond to 0.0405, 0.0320 and 0.1476 percentage points. The CPU instrumented/original raw-head discrepancy was zero. The study is exploratory on one position and does not establish a general chess concept.

**M3 - Native bounded search.** `WORK/research/vegam_atlas/data/analyses/b17cbc321220a5b0cd7c1480/search.json`, originating from the user-run job `20261009T171040Z-4da4def1`. Fresh tree, one thread, batch one, Q enabled, 1,024 root visits. It was reused in the CPU study and was not rerun while preparing this book.

**A1 - Textbook arithmetic and figures.** `WORK/research/inside_vegam_book/prepare.py`, `make_extras.py` and `data/facts.json`. These reconstruct the input projection from saved weights and inputs, compute the teaching examples, validate the game's moves and generate plots from saved tensors. They do not run engine matches, train a model or launch new neural inference.

## E.5 Frozen identities

@table identities

A full SHA-256 identifies file bytes, not chess strength. Its purpose is to prevent a later package, export or modified instrumentation path from silently masquerading as this measurement. The source-index and measurement JSON files preserve the complete strings alongside the PDF.

## E.6 Reproduce without using the GPU

The companion folder is `WORK/research/inside_vegam_book`. Its manuscript files are editable text. `data/catalan.pgn` contains the complete legal game; `data/bg7-record.json` contains the start FEN and exact UCI move prefix; `data/measurements.json` preserves the one-position result; `data/facts.json` stores the derived arithmetic and figure metadata. The figures are static images made from these records, not generated illustrations of imaginary activations.

The shared renderer, `visual_style.py`, uses the installed python-chess SVG piece artwork and system librsvg/Cairo. Matplotlib draws the measured charts. Board highlights and arrows are editorial guides derived from legal square coordinates; they are distinct from the colors that encode measured numerical values.

To rebuild the data-derived figures, run `NEXT/.venv/bin/python WORK/research/inside_vegam_book/prepare.py`, then the same interpreter on `make_extras.py`. The first script reads the trusted checkpoint on CPU, verifies its hash, and does small matrix arithmetic against saved traces. It does not call the network forward function. It also validates every game move and the three local rook replies.

To typeset the book, run the bundled Python interpreter on `WORK/research/inside_vegam_book/build_book.py`. That interpreter is recorded in `BUILD.md`. It supplies ReportLab and PDF tooling; the engine environment supplies NumPy, chess and plotting for the preparation stage. These separate environments are intentional and avoid changing the engine's dependencies just to make a document.

Render the resulting PDF with `pdftoppm -png -r 90`, then inspect the images. Text extraction alone cannot detect every layout problem. `validate_book.py` checks document structure, source existence, math receipts, page text and figure references, while the visual review checks the actual printed pages.

## E.7 Reproduce the neural experiment deliberately

A fresh neural run is a different task from rebuilding the PDF. First choose a new receipt directory; verify checkpoint, exported model, binary and config hashes; and replay the exact saved start FEN and move list. Use the native input dump to compare features, move tuples and legal masks before comparing head outputs. A matching visible board is insufficient.

For diagnostic arithmetic, load the checkpoint's EMA, round weights to float16 as specified, then choose the arithmetic device explicitly. Call the original forward function and the instrumented one on identical arrays. Compare raw heads between those two implementations and normalized probabilities against the native path. Record all tolerances before inspecting results. Never relax a failing tolerance just to obtain a passing narrative.

For interventions, save a baseline, apply the named tap alteration, recompute downstream outputs, record both the chosen probability and logit contrast, and restore. A no-op restoration should return to baseline within the expected numerical behavior. For native search, specify thread count, batch size, Q setting, root history, retained-tree policy and visit or clock budget. Store complete UCI output, including verbose root statistics.

These are reproduction instructions, not a claim that further hardware jobs ran. In this workspace, serious new compute remains subject to the user's job-stack approval policy. The finished textbook does not depend on those optional experiments.

## E.8 The complete anchor history

The following is the exact legal SAN prefix to the anchor, reconstructed from the supplied PGN. It ends after 80...Rh7. Replaying it from the ordinary starting position supplies the history that a standalone anchor FEN lacks. A machine-readable UCI prefix is saved in the companion JSON.

@gameprefix

The resulting FEN is `8/7r/2pR3P/3bB2k/p1p1p3/P3P3/1PK5/8 w - - 17 81`. The eight most recent frames used by the input encoder are printed below, newest first. Piece history is only one part of the input: repetition, castling, side and clock semantics still follow the encoder.

@table history

## E.9 Primary background reading

These references support the broader mathematical and interpretability context. The detailed VEGAM equations and numerical claims above are derived from local source and receipts. Results from another model are not treated as results about VEGAM.

**R1.** Vaswani et al., *Attention Is All You Need* (2017). Original scaled dot-product and multihead attention formulation. https://arxiv.org/abs/1706.03762

**R2.** Ba, Kiros and Hinton, *Layer Normalization* (2016). Per-example normalization with learned gain and bias. https://arxiv.org/abs/1607.06450

**R3.** Hendrycks and Gimpel, *Gaussian Error Linear Units (GELUs)* (2016). Definition of the smooth activation used here. https://arxiv.org/abs/1606.08415

**R4.** Loshchilov and Hutter, *Decoupled Weight Decay Regularization* (2017; ICLR 2019). AdamW's separation of decay from adaptive gradient updates. https://arxiv.org/abs/1711.05101

**R5.** *Toy Models of Superposition* (2022). A controlled setting for understanding overlapping features; not evidence that a specific VEGAM unit has a particular meaning. https://arxiv.org/abs/2209.10652

**R6.** Bricken et al., *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (2023). Sparse dictionary features as an alternative to raw neuron coordinates. https://www.transformer-circuits.pub/2023/monosemantic-features/index.html

**R7.** Jain and Wallace, *Attention is not Explanation* (2019). Empirical cautions about equating attention weights with explanation. https://aclanthology.org/N19-1357/

**R8.** *Evidence of Learned Look-Ahead in a Chess-Playing Neural Network* (NeurIPS 2024). A separate chess-network investigation motivating stronger causal tests. https://proceedings.neurips.cc/paper_files/paper/2024/hash/37d9f19150fce07bced2a81fc87d47a6-Abstract-Conference.html

@pagebreak

## E.10 What this edition does and does not settle

The input contract, tensor shapes, implemented transformations, saved outputs and named interventions are inspectable and reproducible. The teaching arithmetic is computed independently and checked. The game diagrams reproduce legal positions. The current launcher and completed training stage have been distinguished from historical documentation.

What remains unsettled is a general semantic account of the learned representation: whether a particular direction or circuit reliably represents rook confinement, how broadly it generalizes, whether it causally mediates appropriate choices across controlled positions, and whether an architectural or search change improves long-run playing strength. Those uncertainties are stated precisely so the next experiment can resolve them. They are not concealed by giving a hidden unit an attractive name.
