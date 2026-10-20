# Training and export lineage

The published model is the personally trained wide epoch-3 network.
`nextchess/model.py` defines the architecture and heads. `raw.py` and
`fast_raw.py` decode Lc0 records; `cloud_data.py` reconstructs history and
streams training examples; `cloud_train.py` implements training and resumable
data state. The code is provided without cloud credentials or provisioning.

The provenance records preserve the final configuration and execution migrations.
The historical main run used PyTorch **2.12.1+cu130**, NumPy **2.5.1**,
BF16 training, global batch **1024**, and seed **731**. It transitioned from
distributed to single-GPU execution while retaining the global batch.
Those migrations were not claimed to be bitwise numerical continuations.

## Data preparation

Obtain compatible Lc0 V6 training archives under the original data source's terms.
The multi-terabyte raw corpus is not included in this repository.

```sh
python3 -m nextchess.index --help
python3 -m nextchess.transport --help
python3 -m nextchess.cloud_train --help
```

The indexer takes an explicit corpus root. Archive identity, split rules,
positions and replayable reader state are part of the pipeline. Inspect the
CLI arguments before starting: indexing and training are substantial jobs.
An exact reproduction of the historical training run also requires its original
archives and manifest, not merely the architecture code.

## Export existing weights

Fetch the EMA-only checkpoint:

```sh
python3 tools/fetch_models.py --ema
```

Install the PyTorch/ONNX development environment separately; this is unnecessary
for playing the prebuilt engine. The released EMA asset holds model/configuration
and training metadata only, with no optimizer, credentials or cloud state.

To reproduce the fallback graph:

```sh
python3 -m nextchess.export models/vegam-epoch3-ema.pt output/raw.onnx --half
python3 tools/optimize_inference_graph.py output/raw.onnx output/model.onnx
```

For the TensorRT source graph:

```sh
python3 tools/export_mixed_precision.py \
  models/vegam-epoch3-ema.pt --metadata-from models/model.onnx \
  --islands residual,gelu,heads output/model.trt-source.onnx
```

The source graph keeps residual streams, normalization, GELU and readout heads
in FP32 while using FP16-rounded learned weights and FP16 large products.
The TensorRT builder uses strong typing and disables TF32.

Export-library changes can alter graph serialization and floating-point
execution. The published assets are the preserved artifacts; matching a
training recipe alone does not establish identical hashes or outputs.
