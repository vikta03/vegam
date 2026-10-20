# Third-party notices

## Leela Chess Zero

The engine is derived from Lc0 v0.32.0 at
`b38ed00a25baed9554d2675ec376bd50dad18195`.
Copyright The LCZero Authors and the individual contributors identified in
`engine/AUTHORS`, source headers and upstream history.
GPL-3.0-or-later, with the additional NVIDIA linking permission retained in
the applicable source files. Full licence: COPYING.

`engine/libs/lczero-common` is the pinned upstream protocol source. Libraries
inside `engine/third_party` and `engine/subprojects` retain their own licences,
including GoogleTest, Eigen, half and other bundled components. Their notices
are preserved in place; inclusion does not claim VEGAM authorship.

## ONNX Runtime

The C/C++ headers under `third_party/onnxruntime/include` are from ONNX Runtime
1.30.0. Copyright Microsoft Corporation. MIT licence, included at
`third_party/onnxruntime/LICENSE`.
Runtime binaries are installed separately from the vendor's distributions.

## Chess-piece artwork

The SVG piece drawings used by python-chess are by **Colin M. L. Burnett**,
licensed **CC BY-SA 3.0**. Source:
https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces
and https://python-chess.readthedocs.io/en/latest/svg.html.

The publication includes color-adjusted SVG/PNG versions in `book/figures/pieces`
and board illustrations incorporating them. These drawings and their adaptations
retain CC BY-SA 3.0; the original surrounding writing is separately CC BY 4.0.
The legal text is included in `third_party/licenses/CC-BY-SA-3.0.txt`.

python-chess itself is GPL-3.0-or-later; copyright Niklas Fiekas and contributors.
It is installed as a dependency, not relicensed here.

## Fonts and vendor runtimes

Book rebuilding uses Caladea (SIL Open Font License) and DejaVu fonts
(their own permissive font licence), installed separately. Fonts embedded in the
PDF retain their original rights.

CUDA, cuDNN and TensorRT libraries are not committed to this repository or
included in the runtime-free engine archive. Installation retrieves official
packages, which carry their own NVIDIA licence agreements and notices.
The model's reuse licence does not apply to those libraries.
