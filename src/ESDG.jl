module ESDG

using Reexport
@reexport using StartUpDG

@reexport using StaticArrays: SVector
@reexport using StructArrays: StructArray,components,foreachfield

using SparseArrays

using LinearAlgebra, MAT # for SBP
using Triangulate, Printf
using RecipesBase, Colors

using CheapThreads
using ThreadingUtilities

export tmap!, resetCheapThreads
@inline function tmap!(f,out,x)
    @batch for i in eachindex(x)
        @inbounds out[i] = f(x[i])
    end
    return out # good practice for mutating functions?
end

# if CheapThreads freezes, this might fix it. Must run manually (not sure how to automatically detect freezes). 
function resetCheapThreads()
    CheapThreads.reset_workers!()
    ThreadingUtilities.reinitialize_tasks!()
end

include("ModalESDG.jl")
export hybridized_SBP_operators, ModalESDG

include("DiagESBP.jl")
include("NodalESDG.jl")
export DiagESummationByParts, NodalESDG

include("flux_differencing.jl")
export hadamard_sum_ATr!

# support for triangular meshes via Triangulate.jl (Jonathan Shewchuk's Triangle software)
include("triangulate_utils.jl")
export triangulateIO_to_VXYEToV, get_node_boundary_tags, refine

# some prebuilt meshes
include("triangulate_example_meshes.jl")
export rectangular_domain, square_domain, square_hole_domain, scramjet 

include("triangulate_plotting.jl")
export MeshPlotter, BoundaryTagPlotter # mesh plot recipes

end
