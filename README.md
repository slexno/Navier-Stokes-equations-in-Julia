diff --git a/README.md b/README.md
index a39a589de985af743c5eb65f6120b013efe727d4..16b47aad46de9fba6b844dfa030a455529fb5523 100644
--- a/README.md
+++ b/README.md
@@ -1,2 +1,22 @@
 # Navier-Stokes-equations-in-Julia
-write in julia a 2D version of a solver of the Navier-Stockes équations.
+Write in Julia a 2D version of a solver for the Navier–Stokes equations.
+
+## Added staggered-grid discretization utilities
+
+The file `staggered_grid.jl` now provides:
+
+- Allocation of MAC-grid fields:
+  - pressure `p`: `(Nx, Ny)`
+  - velocity `u`: `(Nx + 1, Ny)`
+  - velocity `v`: `(Nx, Ny + 1)`
+- Computation of `du/dx` and `dv/dy` at cell centers and interpolation to:
+  - x-normal faces
+  - y-normal faces
+- Computation of `du/dy` and `dv/dx` (for diffusive cross-direction fluxes)
+- Diffusive fluxes for both velocity components:
+  - `Fx_udiff = ν (du/dx) dSx`
+  - `Fy_udiff = ν (du/dy) dSy`
+  - `Fx_vdiff = ν (dv/dx) dSx`
+  - `Fy_vdiff = ν (dv/dy) dSy`
+
+All outputs are returned as arrays located on the corresponding faces.
