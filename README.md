# JuliaDG
## A set of codes for solving 1D/2D/3D time-dependent hyperbolic PDEs using high order DG methods on 1D and 2D (triangular and quadrilateral) unstructured meshes.  This branch accompanies CAAM 542: Discontinuous Galerkin Methods. 

These codes are inspired by the Matlab codes for the book Nodal Discontinuous Galerkin methods by Hesthaven and Warburton (2007).  All demos use simple uniform meshes, and the codebase is intended mainly for experimentation and method development.

## Demos for linear problems
- The simplest demo is "dg1D_advec.jl", which solves the 1D advection equation on a uniform 1D mesh.
