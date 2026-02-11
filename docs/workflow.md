# Workflow and CLI

## CLI (autodetect input formats)

```bash
JAX_ENABLE_X64=1 bimfx --input inputs/knot_tube.csv --normals inputs/knot_tube_normals.csv --method mfs --validate \
  --k-nn 24 --lambda-reg 1e-6 --acceleration barnes-hut
```

Mesh inputs are supported via `trimesh`:

```bash
pip install "bimfx[io]"
JAX_ENABLE_X64=1 bimfx --input my_surface.stl --method bim --estimate-normals
```

## Pipeline config

Use a TOML config to run a full “solve → validate” pipeline:

```bash
python -c "from bimfx.pipeline import run_pipeline; run_pipeline('examples/pipeline.toml')"
```

Example config:

```toml
[boundary]
input = "inputs/knot_tube.csv"
normals = "inputs/knot_tube_normals.csv"
format = "csv"
subsample = 200

[solve]
method = "mfs"
toroidal_flux = 1.0
k_nn = 24
lambda_reg = 1e-6
acceleration = "barnes-hut"
accel_theta = 0.6
accel_leaf_size = 64

[validate]
eps_factor = 0.02

[output]
dir = "outputs/pipeline"
```
