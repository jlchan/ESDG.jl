module ESDG

using NodesAndModes
using StartUpDG

using StructArrays: components,foreachfield

using CheapThreads
using MAT # convert from .mat files 
using Triangulate, Printf

export tmap!
@inline function tmap!(f,out,x)
    @batch for i in eachindex(x)
        @inbounds out[i] = f(x[i])
    end
end


export hadamard_sum_ATr!
include("flux_differencing.jl")

# support for triangular meshes via Triangulate.jl (Jonathan Shewchuk's Triangle software)
export triangulateIO_to_VXYEToV,get_node_boundary_tags,
       plot_mesh,plot_mesh!,plot_segment_tags,plot_segment_tags!
include("triangulate_utils.jl")

export rectangular_domain,square_domain,square_hole_domain,scramjet,refine
include("triangulate_example_meshes.jl")

end
