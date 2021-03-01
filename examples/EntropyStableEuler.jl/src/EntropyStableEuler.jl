"""
    Module EntropyStableEuler

Includes general math tools
"""

module EntropyStableEuler
using StaticArrays

const γ = 1.4

export logmean, γ
include("./logmean.jl")

# # submodules
# export Fluxes1D,Fluxes2D,Fluxes3D

# export entropy_scaling,scale_entropy_input,scale_entropy_output
# changes definition of entropy variables by a constant scaling
#const entropy_scaling = 1/γ # implies vE = -1 /(ι * γ) and γι = non-dim temperature if R* = cp
const entropy_scaling = 1
scale_entropy_output(V...) = (x->@. x*entropy_scaling).(V)
scale_entropy_input(V...) = (x->@. x/entropy_scaling).(V)

#####
##### one-dimensional fluxes
#####

module Fluxes1D
import ..γ
import ..entropy_scaling
import ..EntropyStableEuler: logmean
import ..EntropyStableEuler: scale_entropy_output, scale_entropy_input

export wavespeed_1D

export primitive_to_conservative, conservative_to_primitive_beta
export u_vfun, v_ufun
export euler_fluxes
export Sfun,pfun,betafun
include("./euler_fluxes_1D.jl")
include("./entropy_variables.jl")
end

#####
##### two-dimensional fluxes
#####
module Fluxes2D
import ..γ
import ..entropy_scaling
import ..EntropyStableEuler: logmean, entropy_scaling
import ..EntropyStableEuler: scale_entropy_output, scale_entropy_input

export primitive_to_conservative,conservative_to_primitive_beta
export u_vfun, v_ufun
export Sfun,pfun,betafun
export euler_fluxes

include("./entropy_variables.jl")
include("./euler_fluxes_2D.jl")
end

#####
##### three-dimensional fluxes
#####
module Fluxes3D
import ..γ
import ..entropy_scaling
import ..EntropyStableEuler: logmean, entropy_scaling
import ..EntropyStableEuler: scale_entropy_output, scale_entropy_input

export primitive_to_conservative,conservative_to_primitive_beta
export u_vfun, v_ufun
export Sfun,pfun,betafun
export euler_fluxes

include("./entropy_variables.jl")
include("./euler_fluxes_3D.jl")
end

# export u_vfun, v_ufun, betafun, pfun, rhoe_ufun
# export dVdU_explicit, dUdV_explicit
# export wavespeed # c
# export Sfun, sfun # math/physical entropies
# export u_vfun1D, v_ufun1D, betafun1D # specialization to 1D
# export primitive_to_conservative
# include("./euler_variables.jl")

# export euler_fluxes
# export euler_flux_x, euler_flux_y # separate x,y fluxes for faster implicit assembly using ForwardDiff
# include("./euler_fluxes.jl")

# export vortex
# include("./analytic_solutions.jl")

end
