include("vis_utils.jl")

# inputs = plotting points only. This should be the most general version

# call example: mesh(Vp*u,(x->Vp*x).(rd.rst),(x->Vp*x).(md.xyz))
function Makie.convert_arguments(P::Type{<:Makie.Mesh},uplot,rst_plot,xyz_plot)

    t = permutedims(plotting_triangulation(rst_plot))
    makie_triangles = Makie.to_triangles(t)

    num_elements = size(first(xyz_plot),2)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,num_elements)
    coordinates = zeros(length(first(rst_plot)),3)
    for e = 1:num_elements       
        for i = 1:2
            coordinates[:,i] .= view(xyz_plot[i],:,e)
        end
        coordinates[:,3] .= view(uplot,:,e)
        trimesh[e] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles) # speed this up?
    end
    return tuple(merge([trimesh...]))
end

# specify interpolation matrix = pass in nodal values for everything else. 
# call via Makie.mesh(u,Vp,rd.rst,md.xyz, color=vec(Vp*u)). 
function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,Vp::Matrix,rst,xyz)
    convert_arguments(P,Vp*u,(x->Vp*x).(rst),(x->Vp*x).(xyz))
end

# specify Nplot 
function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,Nplot::Int,rd::RefElemData,md::MeshData)
    Vp = plotting_interpolation_matrix(Nplot,rd)
    return convert_arguments(P,u,Vp,rd.rst,md.xyz)
end

# specify nothing, use rd.Vp for plot interp matrix
function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,rd::RefElemData,md::MeshData)
    return convert_arguments(P,u,rd.Vp,rd.rst,md.xyz)
end


