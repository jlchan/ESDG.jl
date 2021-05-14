module ESDG

using StartUpDG

using CheapThreads
using MAT # convert from .mat files 
using Triangulate, Printf

using Triangulate, Printf
using Plots: plot,plot!
import Triangulate:triangulate
export triangulateIO_to_VXYEToV,get_node_boundary_tags,
       plot_mesh,plot_mesh!,plot_segment_tags,plot_segment_tags!
include("triangulate_utils.jl")
export rectangular_domain,square_domain,square_hole_domain,scramjet,refine
include("triangulate_example_meshes.jl")

end
