include("vis_utils.jl")

# inputs = plotting points only. most general
# call example:s mesh(Vp*u,(x->Vp*x).(rd.rst),(x->Vp*x).(md.xyz))
function Makie.convert_arguments(P::Type{<:Makie.Mesh},uplot,rst_plot,xyz_plot)
    # build reference triangulation
    rp,sp = rst_plot
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims([rp sp]) 
    triout,_ = triangulate("Q", triin)
    t = permutedims(triout.trianglelist) 
    makie_triangles = Makie.to_triangles(t)

    num_elements = size(first(xyz_plot),2)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,num_elements)
    coordinates = zeros(length(rp),3)
    for e = 1:num_elements       
        for i = 1:2
            coordinates[:,i] .= view(xyz_plot[i],:,e)
        end
        coordinates[:,3] .= view(uplot,:,e)
        trimesh[e] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles) # speed this up?
    end
    return tuple(merge([trimesh...]))
end

function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,rd::RefElemData,md::MeshData)
    return convert_arguments(P,u,rd.Vp,rd.rst,md.xyz)
end

function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,Nplot::Int,rd::RefElemData,md::MeshData)
    Vp = plotting_interpolation_matrix(Nplot,rd)
    return convert_arguments(P,u,Vp,rd.rst,md.xyz)
end

# specify interpolation matrix = pass in nodal values for everything else
# call via Makie.mesh(u,Vp,rd.rst,md.xyz, color=vec(Vp*u))
function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,Vp::Matrix,rst,xyz)

    # build reference triangulation
    rp,sp = (x->Vp*x).(rst)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims([rp sp]) 
    triout,_ = triangulate("Q", triin)
    t = permutedims(triout.trianglelist) 
    makie_triangles = Makie.to_triangles(t)

    num_elements = size(first(xyz),2)
    trimesh = Vector{GeometryBasics.Mesh{3,Float32}}(undef,num_elements)
    coordinates = zeros(length(rp),3)
    for e = 1:num_elements       
        xyze = view.(xyz,:,e)
        for d = 1:length(xyze)
            mul!(view(coordinates,:,d),Vp,xyze[d])
        end
        mul!(view(coordinates,:,3),Vp,view(u,:,e))
        trimesh[e] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),makie_triangles) # speed this up?
    end
    return tuple(merge([trimesh...]))
end

