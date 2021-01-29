# ESDG.jl (**e**nergy/**e**ntropy stable **d**iscontinuous **G**alerkin)
## A Julia codebase for solving 1D/2D/3D time-dependent hyperbolic PDEs using energy or entropy stable high order DG methods on conforming unstructured meshes consisting of triangular, quadrilateral, or hexahedral elements.

These codes are inspired by the Matlab codes for the book [Nodal Discontinuous Galerkin methods](https://link.springer.com/book/10.1007/978-0-387-72067-8) by Hesthaven and Warburton (2007).  While unstructured meshes are supported, all demos use uniform meshes, and the codebase is intended mainly for experimentation and method development.

This codebase builds off of [NodesAndModes.jl](https://github.com/jlchan/NodesAndModes.jl) and [StartUpDG.jl](https://github.com/jlchan/StartUpDG.jl). Special thanks to [Yimin Lin](https://github.com/yiminllin) for providing the initial routines which started this codebase.

<!-- - [On discretely entropy stable weight-adjusted DG methods: curvilinear meshes](https://doi.org/10.1016/j.jcp.2018.11.010)-->
<!-- using Pkg
Pkg.add("Revise")
Pkg.add("Plots")
Pkg.add("PyPlot")
Pkg.add("SpecialFunctions")
Pkg.add("Documenter")

?[Module/Function name] for documentation -->
