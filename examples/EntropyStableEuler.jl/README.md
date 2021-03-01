# EntropyStableEuler

[![Build Status](https://travis-ci.com/jlchan/EntropyStableEuler.jl.svg?branch=master)](https://travis-ci.com/jlchan/EntropyStableEuler.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/jlchan/EntropyStableEuler.jl?svg=true)](https://ci.appveyor.com/project/jlchan/EntropyStableEuler-jl)
[![Codecov](https://codecov.io/gh/jlchan/EntropyStableEuler.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jlchan/EntropyStableEuler.jl)

Entropy stable finite volume fluxes and formulas for compressible Euler and Navier-Stokes. Code based off of formulas in [Chandrashekar 2012](https://doi.org/10.4208/cicp.170712.010313a) and [Winters et al. 2019](https://link.springer.com/article/10.1007/s10543-019-00789-w). Formulas for entropy variables are from [Hughes, Mallet, Franca 1986](https://doi.org/10.1016/0045-7825(86)90127-1) and from [Parsani, Carpenter, Nielsen 2015](https://doi.org/10.1016/j.jcp.2015.03.026)

# Usage

The package consists mainly of three sub-modules Fluxes1D, Fluxes2D, Fluxes3D.
Each module exports entropy conservative fluxes, as well as helper routines.

For example, Fluxes2D exports
- `euler_fluxes`, which evaluates entropy conservative fluxes
- `u_vfun, v_ufun` to convert between conservative and entropy variables
- `conservative_to_primitive_beta` to convert between conservative and "primitive" variables (involving inverse temperature β) used to evaluate fluxes.
```
using EntropyStableEuler.Fluxes2D

# construct solution at two states
rhoL,rhouL,rhovL,EL = map(x->x.*ones(4),(1,.1,.2,2))
rhoR,rhouR,rhovR,ER = map(x->x.+.1*randn(4),(rhoL,rhouL,rhovL,EL)) # small perturbation

# convert to "primitive" variables for efficient flux evaluation
rhoL,uL,vL,betaL = conservative_to_primitive_beta(rhoL,rhouL,rhovL,EL)
rhoR,uR,vR,betaR = conservative_to_primitive_beta(rhoR,rhouR,rhovR,ER)

# evaluate fluxes
Fx,Fy = euler_fluxes(rhoL,uL,vL,betaL,rhoR,uR,vR,betaR)

# can also pass in precomputed log values for efficiency
#      euler_fluxes(rhoL,uL,vL,betaL,rhologL,betalogL,
#                   rhoR,uR,vR,betaR,rhologR,betalogR)
```

# To-do
- add Lax-Friedrichs penalty and matrix dissipation from [Winters et al. 2017](https://doi.org/10.1016/j.jcp.2016.12.006)
- add Jacobians for transforms between conservative and entropy variables
