module ESDG

using Reexport
@reexport using StartUpDG
@reexport using StaticArrays: SVector
# @reexport using StructArrays # don't do this - doesn't export master branch?

using Setfield
using LinearAlgebra, SparseArrays 
using MAT # for SBP node reading
using Triangulate, Printf
using DiffEqBase # for callbacks
using Polyester, ThreadingUtilities
using RecipesBase, Colors 
import TriplotRecipes
# using GeometryBasics, Makie

include("ode_utils.jl")
export monitor_callback

include("ModalESDG.jl")
export ModalESDG

include("flux_differencing.jl")
export hadamard_sum_ATr!

# triangular meshes via Triangulate.jl (Jonathan Shewchuk's Triangle software)
include("mesh/triangulate_utils.jl")
export triangulateIO_to_VXYEToV, get_node_boundary_tags, refine

# some prebuilt meshes
include("mesh/triangulate_example_meshes.jl")
export rectangular_domain, square_domain, square_hole_domain, scramjet 

include("visualization/mesh_plotting.jl")
include("visualization/Triplot_support.jl")
# include("visualization/Makie_support.jl")
export plotting_interpolation_matrix, DGTriPseudocolor, MeshPlotter, BoundaryTagPlotter 

include("misc_utils.jl")
export tmap!, resetThreads

end
