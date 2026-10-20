# Source and measurement provenance

The public engine is based on the frozen Fixed3 source snapshot from
28 September 2026, used by the VEGAM-TF3-Q daily-driver package.

Original executable:
`3824cc57b9a7403bd13baabeb69799186271cea24e42c8f1957ae4ff38633482`

Fallback model:
`1534cbc469babebbb6225e0c5890ffeb984fc08e313fd84c78b674974d79e943`

Original training checkpoint:
`79a46c94eac4739abf5b9ae1ba37e4fd4d17a838527d172c63ffae7c12afb6db`

The release rebuild has a new binary hash. Build settings and dependency paths
change for portability; the original search/evaluator C++ files remain frozen.
`tools/check_publication.py` verifies their hashes against the import manifest.

- `import-manifest.json`: explicit imported file inventory and pre-edit hashes.
- `fixed3-source.json`: original engine source receipt.
- `daily-driver/manifest.json`: historical package identity and defaults.
- `upstream.json`: comparison with pristine Lc0 source.
- `training-model.json`, `training-result.json`: final training metadata.
- Root `models.json`: downloadable model identities.
- `release-verification.json`: clean build, retained regressions, provider and
  relocation results for the shipped executable.
- `upstream-status.json`: public GitHub status responses at publication preparation.
- `weights.json`: EMA tensor identity and metadata normalization. The downloadable
  checkpoint supports PyTorch's restricted `weights_only=True` loading.

The historical prefixes `WORK`, `NEXT` and `CORPUS` identify private source
trees, not public paths to open. Original records stay in the author's archive.
Publication copies can differ by path redaction or author-metadata edits;
original hashes are retained as provenance, not as checksums of edited copies.

No old repository history is rewritten. This public repository begins with
an attributed source snapshot; original upstream authors and licence notices
remain present.
