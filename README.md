# JuliaDG
## A set of codes for solving 1D/2D/3D time-dependent hyperbolic PDEs using high order DG methods on either triangular or quadrilateral meshes.

These codes are based loosely on the Matlab codes for the book Nodal Discontinuous Galerkin methods by Hesthaven and Warburton (2007).

## Demos for linear problems
- 1D demos to come soon...
- The simplest demo is "dg2D_advec_tri.jl", which simulates the advection equation on a triangular mesh.
- The demos "dg2D_wave_tri.jl" and "dg2D_wave_quad.jl" compute solutions to the acoustic wave equation on triangular and quadrilateral meshes.
- The demo "dg3D_advec_hex.jl" solves the advection equation on a hexahedral mesh.

## Demos for nonlinear problems
- The file "dg2D_burgers_quad.jl" provides a demo of an entropy stable DG method for Burgers' equation on a quadrilateral mesh.
- The file "dg2D_euler_quad.jl" and "dg3D_euler_hex.jl" provide entropy stable DG methods on quadrilateral and hexahedral meshes for the compressible Euler equations, with a sparsity-optimized implementation of the Hadamard sum step in flux differencing.
- More to come...

## References
The discretizations used are based on the following references:
- [On discretely entropy conservative and entropy stable DG methods
](https://doi.org/10.1016/j.jcp.2018.02.033)
- [On discretely entropy stable weight-adjusted DG methods: curvilinear meshes](https://doi.org/10.1016/j.jcp.2018.11.010)
- [Efficient Entropy Stable Gauss Collocation Methods](https://doi.org/10.1137/18M1209234)
- [Skew-Symmetric Entropy Stable Modal DG Formulations](https://doi.org/10.1007/s10915-019-01026-w)

<!-- using Pkg
Pkg.add("Revise")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("SpecialFunctions")
Pkg.add("Documenter")

?[Module/Function name] for documentation -->
