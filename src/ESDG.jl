module ESDG

using Reexport
@reexport using NodesAndModes
@reexport using StartUpDG

using StaticArrays
using StructArrays
using StructArrays: components,foreachfield

using LinearAlgebra
using MAT # read from .mat files 
using Triangulate, Printf
using RecipesBase
using ColorTypes:HSV

using CheapThreads

export tmap!
@inline function tmap!(f,out,x)
    @batch for i in eachindex(x)
        @inbounds out[i] = f(x[i])
    end
end

include("ModalESDG.jl")
export hybridized_SBP_operators,ModalESDG

include("DiagESBP.jl")
include("NodalESDG.jl")
export DiagESummationByParts,NodalESDG

include("flux_differencing.jl")
export hadamard_sum_ATr!

# support for triangular meshes via Triangulate.jl (Jonathan Shewchuk's Triangle software)
include("triangulate_utils.jl")
export triangulateIO_to_VXYEToV,get_node_boundary_tags,
       plot_mesh,plot_mesh!,plot_segment_tags,plot_segment_tags!

# convenience constructs for some simple meshes
include("triangulate_example_meshes.jl")
export rectangular_domain,square_domain,square_hole_domain,scramjet,refine

include("triangulate_plotting.jl")
export MeshPlotter,BoundaryTagPlotter

end
