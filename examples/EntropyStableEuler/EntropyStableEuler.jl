"""
    Module EntropyStableEuler

Includes general math tools
"""

module EntropyStableEuler
using StaticArrays

const γ=1.4
export γ
export logmean
export u_vfun, v_ufun, betafun, pfun, rhoe_ufun
export dVdU_explicit, dUdV_explicit
export wavespeed # c
export Sfun, sfun # math/physical entropies
export u_vfun1D, v_ufun1D, betafun1D # specialization to 1D
export euler_fluxes, euler_flux_x, euler_flux_y # separate x,y flux for faster implicit assembly using ForwardDiff
export vortex, primitive_to_conservative

include("./logmean.jl")
include("./euler_fluxes.jl")
include("./euler_variables.jl")

# 2D isentropic vortex solution for testing. assumes domain around [0,20]x[-5,5]
function vortex(x,y,t)

    x0 = 5
    y0 = 0
    beta = 5
    r2 = @. (x-x0-t)^2 + (y-y0)^2

    u = @. 1 - beta*exp(1-r2)*(y-y0)/(2*pi)
    v = @. beta*exp(1-r2)*(x-x0-t)/(2*pi)
    rho = @. 1 - (1/(8*γ*pi^2))*(γ-1)/2*(beta*exp(1-r2))^2
    rho = @. rho^(1/(γ-1))
    p = @. rho^γ

    return (rho, u, v, p)
end

end
