# Inside VEGAM
## From Chess Pieces to Learned Intelligence

By **Remmy Victor**.

[Download the illustrated book](https://github.com/vikta03/vegam/releases/download/v1.0.0/inside-vegam.pdf)
or [watch the silent bishop reel](https://github.com/vikta03/vegam/releases/download/v1.0.0/inside-vegam-bishop-reel.mp4).

The 102-page edition contains 15 chapters, five appendices, 31 figures,
23 displayed equation groups and 24 exercises with answers. Its measurements
are a frozen 9 October 2026 snapshot. Publication changes credits and portable
paths without claiming new inference results.

![A move, viewed through the model](figures/g7_layers.png)

The central example is **81.Bg7** in the supplied Catalan game. The book walks
from pieces and history into square embeddings, attention, adaptive relations,
MLPs, the move decoder and search. Additional positions help distinguish a
repeatable mechanism from a story about one diagram.

## Rebuild the document

Install the repository's `requirements-book.txt` and the Caladea/DejaVu system
fonts. On Debian/Ubuntu the font packages are `fonts-crosextra-caladea` and
`fonts-dejavu-core`. Alternatively set `VEGAM_FONT_DIR` to a directory containing
the matching TTF files.

```sh
python3 book/build_book.py
```

This uses the supplied figures and frozen measurements; it requires no engine,
GPU or training. Output: `output/pdf/inside-vegam-from-chess-pieces-to-learned-intelligence.pdf`.

The editable text is in `manuscript/`. Numerical receipts, the game and worked
arithmetic are in `data/`. `inputs/` contains saved trace arrays and selected
position summaries used to derive the figures.

To rederive figures, first fetch the EMA-only model with
`python3 tools/fetch_models.py --ema` and install PyTorch, Matplotlib and
python-chess. Then run `book/prepare.py` and `book/make_extras.py` before rebuilding.
The included piece PNGs avoid a rasterizer dependency; regenerating those
PNGs requires system librsvg, Cairo and PyGObject.

`data/source-index.json` preserves historical source identifiers and hashes.
`data/publication-source-index.json` identifies the corresponding included
publication sources. A hash of a redacted receipt is not represented as the
hash of its original local file.

Original writing: CC BY 4.0. Third-party piece artwork retains CC BY-SA 3.0;
see [the notices](../THIRD_PARTY.md).
