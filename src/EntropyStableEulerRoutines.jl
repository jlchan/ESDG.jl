"""
    Module EntropyStableEulerRoutines

Includes general math tools
"""

module EntropyStableEulerRoutines

export logmean
export UV, VU, betafun
export euler_fluxes

using SpecialFunctions

include("./EntropyStableEuler/logmean.jl")
include("./EntropyStableEuler/euler_fluxes.jl")
include("./EntropyStableEuler/euler_variables.jl")

end
