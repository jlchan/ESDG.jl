"""
    Module Utils

General purpose utilities usable by all element types

"""

module Utils

export meshgrid
export geometric_factors_2D
export connect_mesh
export build_node_maps, build_periodic_boundary_maps

include("./Utils/meshgrid.jl")
include("./Utils/geometric_factors_2D.jl")
include("./Utils/connect_mesh.jl")
include("./Utils/build_node_maps.jl")
include("./Utils/build_periodic_boundary_maps.jl")

end
