# Release verification — 20 October 2026

J31 built the frozen source with GCC 16.2.1 on Fedora 44, using two CPU cores.
The complete run and the reviewed packaging correction used 221.24 seconds
of supervised wall time. Peak summed host RSS was 3.92 GiB.
No training, games, strength match or benchmark sweep was run.

The retained C++ harness passed 122,356 assertions over 27 comparator states,
16 histories, 160 history prefixes and 54 board-identity pairs.

The final executable passed UCI startup/readiness, its shipped Q-head settings,
short legal searches, immediate mate, expanded terminal repetition, stop and
clean quit under:

- Explicit CPU, CUDA and TensorRT providers.
- Automatic CPU fallback with GPU provider libraries absent.
- A launcher alias and working/package directories containing spaces.

The installed executable has only `$ORIGIN/lib` in its runtime search path.
The initial build's temporary absolute ONNX-library path was found during final
inspection and removed by using Meson's staged installation. The final executable
was tested again after that correction. No search or evaluation C++ changed.

The public TensorRT preparation command produced a new local plan from the
hash-pinned mixed-precision graph. On the saved eight-frame Bg7 example, the
largest probability difference from the daily driver's plan was 0.000000183
(the action-WDL output); the duration difference was below 0.000001.
This is a one-position packaging comparison, not a global precision proof.

Runtime libraries were reused from the installed, pinned vendor versions.
The complete multi-gigabyte runtime download was not repeated. The public
installer and dependency pins are supplied; this run is not a fresh operating
system installation test or a cross-distribution certification.

The prebuilt executable requires glibc 2.43, GLIBCXX_3.4.32 and CXXABI_1.3.15.
Use a local source build on older Linux distributions.

Machine-readable results: [release-verification.json](../provenance/release-verification.json).
Historical performance and match evidence: [research](../research/README.md).
