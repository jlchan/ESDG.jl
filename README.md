# JuliaDG
## A set of codes for solving 2D time-dependent hyperbolic PDEs using high order DG methods on either triangular or quadrilateral meshes.

These codes are based loosely on the Matlab codes for the book Nodal Discontinuous Galerkin methods by Hesthaven and Warburton (2007).

The file "dg2D_burgers_quad.jl" provides a demo of entropy stable DG methods on quadrilateral elements for Burgers' equation.

The routines "dg2D_wave_tri.jl" and "dg2D_wave_quad.jl" compute solutions to the acoustic wave equation on triangular and quadrilateral meshes.

The methods used are based on the following references:
- [On discretely entropy conservative and entropy stable DG methods
](https://doi.org/10.1016/j.jcp.2018.02.033)
- [On discretely entropy stable weight-adjusted DG methods: curvilinear meshes](https://doi.org/10.1016/j.jcp.2018.11.010)
- [Efficient Entropy Stable Gauss Collocation Methods](https://doi.org/10.1137/18M1209234)
- [Skew-Symmetric Entropy Stable Modal DG Formulations](https://doi.org/10.1007/s10915-019-01026-w)

Currently, JuliaDG uses the following non-base packages:
- Revise
- Documenter

<!-- using Pkg
Pkg.add("Revise")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("SpecialFunctions")
Pkg.add("Documenter")

?[Module/Function name] for documentation -->
