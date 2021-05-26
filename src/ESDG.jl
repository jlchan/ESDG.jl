module ESDG

using Reexport
@reexport using StartUpDG
# @reexport using StaticArrays
# @reexport using StructArrays # don't do this - doesn't export master branch?

using LinearAlgebra, SparseArrays 
using MAT # for SBP node reading
using Triangulate, Printf
using DiffEqBase # for callbacks
using Polyester, ThreadingUtilities
using RecipesBase, Colors, TriplotRecipes
using GeometryBasics, Makie 

include("ode_utils.jl")
export monitor_callback

include("visualization/Triplot_support.jl")
export plotting_interpolation_matrix, DGTriPseudocolor

include("ModalESDG.jl")
export hybridized_SBP_operators, ModalESDG

include("DiagESBP.jl")
export DiagESummationByParts, DiagESummationByParts!

include("NodalESDG.jl")
export NodalESDG

include("flux_differencing.jl")
export hadamard_sum_ATr!

# triangular meshes via Triangulate.jl (Jonathan Shewchuk's Triangle software)
include("mesh/triangulate_utils.jl")
export triangulateIO_to_VXYEToV, get_node_boundary_tags, refine

# some prebuilt meshes
include("mesh/triangulate_example_meshes.jl")
export rectangular_domain, square_domain, square_hole_domain, scramjet 

include("mesh/triangulate_plotting.jl")
export MeshPlotter, BoundaryTagPlotter # mesh plot recipes

include("misc_utils.jl")
export tmap!, resetThreads

end
