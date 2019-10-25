"""
    Module EntropyStableEuler

Includes general math tools
"""

module EntropyStableEuler

const γ=1.4
export logmean
export u_vfun, v_ufun, betafun
export euler_fluxes, wavespeed
export vortex

using SpecialFunctions

include("./logmean.jl")
include("./euler_fluxes.jl")
include("./euler_variables.jl")
include("./vortex.jl")
end
