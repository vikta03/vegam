# Play VEGAM on Linux

## Requirements

- Linux x86-64. The prebuilt v1.0.0 binary was built on Fedora 44 and requires
  **glibc 2.43**, **GLIBCXX_3.4.32** and **CXXABI_1.3.15** runtime symbols.
  On older distributions, use the [source-build instructions](BUILD.md).
- Python **3.11–3.13** and pip for GPU setup (the pinned TensorRT bindings do
  not support Python 3.14). CPU-only installation also supports Python 3.14.
- For GPU operation: an NVIDIA GPU and a driver supporting CUDA 13.
  This recipe pins ONNX Runtime 1.30.0, TensorRT 10.16.1.11 and CUDA 13 libraries.
- A UCI chess GUI, such as Cute Chess.

CPU operation is supported as a fallback, with much lower expected throughput.
Windows and macOS builds are not part of this release.

## Install

Download and extract `vegam-linux-x86_64.tar.gz` from the
[release](https://github.com/vikta03/vegam/releases/tag/v1.0.0).
Verify its SHA-256 against `SHA256SUMS` on that release.
The engine archive includes the fallback model but excludes the large vendor runtimes.

From the extracted directory:

```sh
python3 tools/install_runtime.py --destination . --gpu
```

This downloads a large runtime into `_runtime/` and creates relative library
links in `lib/`. It does not install system packages. Retain both directories
when moving the package. For CPU only, omit `--gpu`; use a fresh package if
switching runtime installations.

If your distribution disables pip outside virtual environments, run that
installer from a virtual environment: its `--target` still installs the libraries
inside the engine directory.

## Add to your GUI

| Field | Value |
|---|---|
| Name | VEGAM-TF3-Q |
| Command | Absolute path to the extracted `vegamt-tf3-q` launcher |
| Working directory | The extracted directory |
| Protocol | UCI |
| Arguments / init strings | Empty |

Use a new GUI entry so old saved UCI options do not override the shipped defaults.
The launcher resolves its location, including symlink aliases, and supplies explicit
model/configuration paths. Paths containing spaces are supported.

Expected settings are two search threads, minibatch eight, 512 MiB search limit,
action values on, moves-left utility off and outcome blending off.

## Prepare TensorRT for your GPU

CUDA inference can run without this step. The faster TensorRT path needs a
serialized plan built on the recipient's hardware. Preparation can take minutes
and uses the GPU; start it explicitly when that is convenient.

```sh
python3 tools/fetch_models.py --destination . --trt
PYTHONPATH=./_runtime LD_LIBRARY_PATH=./lib python3 tools/prepare_tensorrt.py --package .
```

The preparation tool uses the mixed-precision source graph and preserves the
FP32 residual, normalization, GELU and head operations used in the daily driver.
It does not train or change weights.

On startup, inspect `info string next backend ...`. A prepared GPU installation
should report `TensorRT engine` and two inference lanes. The automatic provider
order tries an adjacent TensorRT wrapper, then CUDA, then CPU. Startup messages
state which provider actually loaded.

To require a provider explicitly:

```sh
NEXT_PROVIDER=cpu ./vegamt-tf3-q
NEXT_PROVIDER=cuda ./vegamt-tf3-q
NEXT_PROVIDER=tensorrt ./vegamt-tf3-q
```

An explicitly requested unavailable GPU provider fails rather than silently
claiming acceleration. After changing model files, GPU hardware or runtime
versions, use a fresh package and rebuild the TensorRT plan.
See [NVIDIA's portability rules](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).

## Troubleshooting

- **Missing shared library:** run the runtime installer in the extracted package;
  launch through the script, not `next-chess` directly.
- **CPU unexpectedly selected:** read startup diagnostics; check the NVIDIA
  driver, installed runtime, and whether the TensorRT plan matches this machine.
- **Lower NPS:** provider, position, power limits and competing GPU workloads
  all matter. Compare the same position and time control.
- **Different options in the GUI:** delete stale saved options or create a new entry.
- **Older glibc/libstdc++:** build locally with the supplied source. A local
  verification run on one machine is not a cross-distribution certification.
