"""
    Module EntropyStableEuler

Includes general math tools
"""

module EntropyStableEuler

export logmean
export UV, VU, betafun
export euler_fluxes

using SpecialFunctions

include("./logmean.jl")
include("./euler_fluxes.jl")
include("./euler_variables.jl")

end
