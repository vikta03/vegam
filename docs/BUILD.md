# Build and verify

The source in `engine/` is the corrected Fixed3 snapshot behind the daily driver.
The publication changes its build dependency paths, not the search algorithm.

Install a C++20 compiler, zlib development headers, Python 3.11–3.13, pip, Meson and
Ninja. On Debian/Ubuntu the system packages are typically `build-essential`,
`zlib1g-dev`, `python3-venv`, `meson` and `ninja-build`.

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
python3 tools/install_runtime.py --destination runtime --gpu
python3 tools/fetch_models.py --trt
python3 tools/build.py
```

The build uses two compiler processes by default, disables `-march=native`,
disables LTO, and performs no Meson dependency downloads. Pinned Lc0 common
sources and ONNX Runtime headers are included.

The package appears at `dist/vegam-linux-x86_64/`. Its runtime must be installed
there with `tools/install_runtime.py --destination . --gpu` before moving it
to a GUI. The repository's `runtime/` is a build dependency, not a hidden
absolute runtime link in the produced executable.

Advanced options: `tools/build.py --help` supports explicit model, output,
build-directory and ONNX library paths. It rejects a model whose checksum
differs from the published network.

## Verification

```sh
LD_LIBRARY_PATH="$PWD/runtime/lib" python3 tests/audit_checks.py
python3 tests/uci_release.py --package dist/vegam-linux-x86_64 --provider cpu
python3 tests/uci_release.py --package dist/vegam-linux-x86_64 --provider tensorrt
```

The C++ harness extracts the actual production comparator and links against
the build objects. It checks ordering on real node objects, repetition identity,
legal versus unusable en-passant targets, and supplied legal histories.
The UCI suite covers legal search output, expanded terminal/repetition children,
explicit model identity, stop/quit and relocation. A short suite is a release
correctness check, not a strength match.

`tools/check_publication.py` performs static source, path, credit, manifest and
publication checks. It does not run the engine or use the GPU.

## Release construction

`python3 tools/package_release.py --package dist/vegam-linux-x86_64 --assets ASSET_DIR --verification provenance/release-verification.json`
creates the binary archive, corresponding source archive and SHA-256 list.
Vendor runtimes, caches and locally generated TensorRT plans are excluded.
The source archive includes the modified source and all required build files.
The release verifier records the actual OS, compiler, model hashes and test results.
