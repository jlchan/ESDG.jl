include("vis_utils.jl")

# inputs = plotting points only. This should be the most general version

# call example: mesh(Vp*u,(x->Vp*x).(rd.rst),(x->Vp*x).(md.xyz))
function Makie.convert_arguments(P::Type{<:Makie.Mesh},uplot,rst_plot,xyz_plot)
    # # build reference triangulation
    # triin = Triangulate.TriangulateIO()
    # triin.pointlist = permutedims([rp sp]) 
    # triout,_ = triangulate("cQ", triin)
    # t = permutedims(triout.trianglelist)

    # # remove zero-volume triangles
    # has_volume = fill(true,size(t,1))
    # volume = zeros(size(t,1))
    # for i = 1:size(t,1)
    #     # randomly assign different z-values to triangle for volume calc
    #     v1 = GeometryBasics.Point(rp[t[i,1]],sp[t[i,1]],1.0) 
    #     v2 = GeometryBasics.Point(rp[t[i,2]],sp[t[i,2]],2.0)
    #     v3 = GeometryBasics.Point(rp[t[i,3]],sp[t[i,3]],3.0)
    #     volume[i] = abs(GeometryBasics.volume(Triangle(v1,v2,v3)))
    #     if abs(GeometryBasics.volume(Triangle(v1,v2,v3))) < 50*eps()
    #         has_volume[i] = false
    #     end
    # end
    # t = t[findall(has_volume),:]
    t = permutedims(plotting_triangulation(rst_plot))
    makie_triangles = Makie.to_triangles(t)

    # ref_elem_mesh = GeometryBasics.normal_mesh(Makie.to_vertices([rp sp zero.(rp)]),makie_triangles) 

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

function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,Nplot::Int,rd::RefElemData,md::MeshData)
    Vp = plotting_interpolation_matrix(Nplot,rd)
    return convert_arguments(P,u,Vp,rd.rst,md.xyz)
end

function Makie.convert_arguments(P::Type{<:Makie.Mesh},u,rd::RefElemData,md::MeshData)
    return convert_arguments(P,u,rd.Vp,rd.rst,md.xyz)
end


