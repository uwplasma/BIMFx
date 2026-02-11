API
===

Public interface
----------------

.. autosummary::
   :toctree: _generated

   bimfx.solve_mfs
   bimfx.solve_bim
   bimfx.vacuum.VacuumField
   bimfx.io.BoundaryData
   bimfx.io.load_boundary_csv
   bimfx.io.save_boundary_csv
   bimfx.io.load_boundary
   bimfx.io.estimate_normals
   bimfx.io.boundary_from_vmec_wout
   bimfx.io.boundary_from_slam_npz
   bimfx.io.boundary_from_sflm_npy
   bimfx.io.boundary_from_mesh
   bimfx.io.boundary_from_stl
   bimfx.tracing.trace_fieldlines_rk4
   bimfx.tracing.trace_fieldlines_rk4_jax
   bimfx.solve_mfs_jax
   bimfx.solve_bim_jax
   bimfx.jax_solvers.MFSJaxField
   bimfx.jax_solvers.BIMJaxField
   bimfx.objectives.boundary_residual_objective
   bimfx.objectives.divergence_objective
   bimfx.pipeline.run_pipeline
   bimfx.pipeline.PipelineResult
   bimfx.tracing.poincare_sections
   bimfx.tracing.FieldlineTrace
   bimfx.tracing.PoincareSection
   bimfx.fci.solve_flux_psi_fci
   bimfx.fci.field_alignment_error
   bimfx.fci.extract_isosurfaces
   bimfx.fci.sample_psi_along_fieldlines
   bimfx.fci.fit_flux_surfaces
   bimfx.fci.seed_from_boundary
   bimfx.fci.compute_psi_rz_slices
   bimfx.fci.analyze_flux_surfaces
   bimfx.fci.FluxSurface
   bimfx.fci.PsiSlice
   bimfx.fci.FluxSurfaceAnalysis
   bimfx.fci.plot_psi_along_fieldlines
   bimfx.fci.plot_poincare_with_psi_contours
   bimfx.fci.plot_poincare_overlays
   bimfx.fci.FCISolution
   bimfx.validation.boundary_normal_residual
   bimfx.validation.offset_points_inward
   bimfx.validation.relative_boundary_residual
   bimfx.validation.divergence_on_grid
   bimfx.validation.summary_stats
   bimfx.validation.validate_vacuum_field

Utilities
---------

.. autosummary::
   :toctree: _generated

   bimfx.fieldline.trace_fieldline_rk4
   bimfx.utils.MFSKernelCache
