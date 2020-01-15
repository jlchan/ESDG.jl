# JuliaDG
## A set of codes for solving 1D/2D time-dependent hyperbolic PDEs using high order DG methods on triangular and quadrilateral unstructured meshes.  This branch of the code accompanies the Rice University Spring 2020 course "CAAM 542: Discontinuous Galerkin Methods".

These codes are inspired by the Matlab codes for the book Nodal Discontinuous Galerkin methods by Hesthaven and Warburton (2007).  While unstructured meshes are supported, all demos use uniform meshes, and the codebase is intended mainly for experimentation and method development.

## Demos for linear problems
- The simplest demo is "dg1D_advec.jl", which solves the 1D advection equation on a uniform 1D mesh.
