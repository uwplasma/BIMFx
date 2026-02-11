# Examples

These scripts are designed to be runnable end-to-end and demonstrate common workflows.

## Vacuum solves

- CSV boundary (choose MFS/BIM):
  [examples/solve_vacuum_from_csv.py](https://github.com/uwplasma/BIMFx/blob/main/examples/solve_vacuum_from_csv.py)
- VMEC `wout*.nc` boundary:
  [examples/solve_from_vmec_wout.py](https://github.com/uwplasma/BIMFx/blob/main/examples/solve_from_vmec_wout.py)
- Compare MFS vs BIM across datasets:
  [examples/compare_mfs_bim_inputs.py](https://github.com/uwplasma/BIMFx/blob/main/examples/compare_mfs_bim_inputs.py)
- Pipeline config example:
  [examples/pipeline.toml](https://github.com/uwplasma/BIMFx/blob/main/examples/pipeline.toml)

## FCI + field-line tools

- FCI solve demo:
  [examples/solve_fci_demo.py](https://github.com/uwplasma/BIMFx/blob/main/examples/solve_fci_demo.py)
- Trace and Poincare:
  [examples/trace_and_poincare_demo.py](https://github.com/uwplasma/BIMFx/blob/main/examples/trace_and_poincare_demo.py)
- Flux-surface fitting:
  [examples/fit_flux_surfaces_demo.py](https://github.com/uwplasma/BIMFx/blob/main/examples/fit_flux_surfaces_demo.py)
- Flux-surface analysis:
  [examples/analyze_flux_surfaces_demo.py](https://github.com/uwplasma/BIMFx/blob/main/examples/analyze_flux_surfaces_demo.py)

## Validation

- Validation report for a toroidal dataset:
  [examples/validate_vacuum_solver.py](https://github.com/uwplasma/BIMFx/blob/main/examples/validate_vacuum_solver.py)
- Full validation report (sweeps across datasets):
  [examples/validation_report.py](https://github.com/uwplasma/BIMFx/blob/main/examples/validation_report.py)
  - Optional `--coarse-to-fine` for parameter tuning sweeps.

## Figures

- Generate gallery figures (boundary clouds, field lines, Poincaré, BIM vs MFS):
  [examples/generate_gallery_figures.py](https://github.com/uwplasma/BIMFx/blob/main/examples/generate_gallery_figures.py)

## Notebooks

- Quickstart notebook:
  [notebooks/quickstart.ipynb](https://github.com/uwplasma/BIMFx/blob/main/notebooks/quickstart.ipynb)
