# JuliaDG
## A set of codes for solving 1D/2D/3D time-dependent hyperbolic PDEs using high order DG methods on conforming unstructured meshes of triangular, quadrilateral, or hexahedral elements.

These codes are inspired by the Matlab codes for the book Nodal Discontinuous Galerkin methods by Hesthaven and Warburton (2007).  While unstructured meshes are supported, all demos use uniform meshes, and the codebase is intended mainly for experimentation and method development.

## Demos for linear problems
- The simplest demo is "dg1D_advec.jl", which solves the 1D advection equation on a uniform 1D mesh.
- The demo "dg2D_advec_tri.jl" simulates the advection equation on a triangular mesh.
- The demo "dg2D_wave_quad.jl" computes solutions to the acoustic wave equation on quadrilateral meshes.
- The demo "dg3D_advec_hex.jl" solves the advection equation on a hexahedral mesh.

## Demos for nonlinear problems
- The file "dg2D_euler_quad.jl" and "dg3D_euler_hex.jl" provide entropy stable DG methods on quadrilateral and hexahedral meshes for the compressible Euler equations, with a sparsity-optimized implementation of the Hadamard sum step in flux differencing.

## References
The discretizations used are based on the following references:
- [Nodal discontinuous Galerkin methods](https://link.springer.com/book/10.1007/978-0-387-72067-8)
- [Weight-adjusted discontinuous Galerkin methods: wave propagation in heterogeneous media](https://epubs.siam.org/doi/abs/10.1137/16M1089186?casa_token=j8893ak2KVEAAAAA:wVbmLtWx3ibL03oxn_97uRt7du2cSdf-6XlkHhczsVTmHI_6ndEgHm-fe3W-CmrWKuEf7CEo_i8)
- [On discretely entropy conservative and entropy stable DG methods](https://doi.org/10.1016/j.jcp.2018.02.033)
- [Skew-Symmetric Entropy Stable Modal DG Formulations](https://doi.org/10.1007/s10915-019-01026-w)
- [Efficient Entropy Stable Gauss Collocation Methods](https://doi.org/10.1137/18M1209234)

Special thanks to [Yimin Lin](mailto:yiminlin@rice.edu) for providing initial routines which produced this codebase.

<!-- - [On discretely entropy stable weight-adjusted DG methods: curvilinear meshes](https://doi.org/10.1016/j.jcp.2018.11.010)-->
<!-- using Pkg
Pkg.add("Revise")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("SpecialFunctions")
Pkg.add("Documenter")

?[Module/Function name] for documentation -->
