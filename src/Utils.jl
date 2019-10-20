"""
    Module Utils

General purpose utilities (not dependent on element type, etc)
"""

module Utils

export meshgrid, geometric_factors_2D, build_node_maps_2D

include("./Utils/meshgrid.jl")
include("./Utils/geometric_factors_2D.jl")
include("./Utils/build_node_maps_2D.jl")

end
