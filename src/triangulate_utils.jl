"""
    function Triangulate.triangulate(triin::TriangulateIO,maxarea,minangle=20)

Convenience routine to avoid writing out @sprintf each time. Returns TriangulateIO object.
"""

function Triangulate.triangulate(triin::TriangulateIO,maxarea,minangle=20)
    angle = @sprintf("%.15f",minangle)
    area  = @sprintf("%.15f",maxarea)
    triout,_ = triangulate("pa$(area)q$(angle)Q", triin)
    return triout
end

"""
    function triangulateIO_to_VXYEToV(triout::TriangulateIO)

Computes VX,VY,EToV from a TriangulateIO object.
"""
function triangulateIO_to_VXYEToV(triout::TriangulateIO)
    VX,VY = (triout.pointlist[i,:] for i = 1:size(triout.pointlist,1))
    EToV = permutedims(triout.trianglelist)
    Base.swapcols!(EToV,2,3) # to match MeshData ordering
    return VX,VY,EToV
end

"""
    function get_boundary_face_labels(triout::TriangulateIO,md::MeshData{2})

Find Triangle segment labels of boundary faces. Returns two arguments:
- boundary_face_tags: tags of faces on the boundary
- boundary_faces: list of faces on the boundary of the domain
"""
function get_boundary_face_labels(triout::TriangulateIO,md::MeshData{2})
    segmentlist = sort(triout.segmentlist,dims=1)
    boundary_faces = findall(vec(md.FToF) .== 1:length(md.FToF))
    boundary_face_tags = zeros(Int,length(boundary_faces))
    for (f,boundary_face) in enumerate(boundary_faces)
        element = (boundary_face - 1) ÷ rd.Nfaces + 1
        face    = (boundary_face - 1) % rd.Nfaces + 1
        vertex_ids = sort(EToV[element,rd.fv[face]])
        tag_id = findfirst(c->view(segmentlist,:,c)==vertex_ids,axes(segmentlist,2))
        boundary_face_tags[f] = triout.segmentmarkerlist[tag_id]
    end
    return boundary_face_tags, boundary_faces
end

"""
    function get_node_boundary_tags(triout::TriangulateIO,md::MeshData{2},rd::RefElemData{2,Tri})

Computes node_tags = Nfp x Nfaces * num_elements array where each entry is a Triangulate tag number.
"""
function get_node_boundary_tags(triout::TriangulateIO,md::MeshData{2},rd::RefElemData{2,Tri})
    boundary_face_tags,boundary_faces = get_boundary_face_labels(triout,md)
    node_tags = zeros(Int,size(md.xf,1)÷rd.Nfaces,md.K*rd.Nfaces) # make Nfp x Nfaces*num_elements
    for (i,boundary_face) in enumerate(boundary_faces)
        node_tags[:,boundary_face] .= boundary_face_tags[i]
    end
    node_tags = reshape(node_tags,size(md.xf)...)
end

# ========= plotting routines for mesh visualization ============

"""
    plot_mesh(VX,VY,EToV)
    plot_mesh!(VX,VY,EToV)
    plot_mesh(triout::TriangulateIO)
    plot_mesh!(triout::TriangulateIO)
    
Plots a mesh given vertices + connectivity or a TriangulateIO object.
"""
function plot_mesh(VX,VY,EToV)
    Plots.plot()
    plot_mesh!(VX,VY,EToV)
end
function plot_mesh!(VX,VY,EToV)
    xmesh = Float64[]
    ymesh = Float64[]
    for vertex_ids in eachrow(EToV)
        ids = vcat(vertex_ids, vertex_ids[1])
        append!(xmesh,[VX[ids];NaN])
        append!(ymesh,[VY[ids];NaN])
    end
    display(Plots.plot!(xmesh,ymesh,linecolor=:black,legend=false,ratio=1,title="$(size(EToV,1)) elements"))
end

plot_mesh(triout::TriangulateIO) = plot_mesh(triangulateIO_to_VXYEToV(triout)...)
plot_mesh!(triout::TriangulateIO) = plot_mesh!(triangulateIO_to_VXYEToV(triout)...)

"""
    plot_segment_tags(triout::TriangulateIO)    
    plot_segment_tags!(triout::TriangulateIO)    

Plot boundary segments in colored according to the tag number. 
"""
function plot_segment_tags(triout::TriangulateIO)    
    Plots.plot()
    plot_segment_tags!(triout) 
end
function plot_segment_tags!(triout::TriangulateIO)    
    tags = unique(triout.segmentmarkerlist)
    num_colors = length(tags)
    colors = range(HSV(0,1,1), stop=HSV(360-360÷num_colors,1,1), length=num_colors)
    xseg = zeros(2,size(triout.segmentlist,2))
    yseg = zeros(2,size(triout.segmentlist,2))
    segcolor = HSV{Float32}[]
    for (col,segment) in enumerate(eachcol(triout.segmentlist))
        xseg[:,col] .= triout.pointlist[1,segment]
        yseg[:,col] .= triout.pointlist[2,segment]
        push!(segcolor,colors[triout.segmentmarkerlist[col]])
    end
    for i = 1:num_colors
        color_ids = findall(triout.segmentmarkerlist .== tags[i])

        # hack to get around issues with multiple legend labels appearing when plotting multiple series
        x_i = vec([xseg[:,color_ids]; fill(NaN,length(color_ids))']) 
        y_i = vec([yseg[:,color_ids]; fill(NaN,length(color_ids))']) 

        Plots.plot!(x_i,y_i,mark=:circle,color=permutedims(segcolor[color_ids]),
                    ratio = 1,label=string(tags[i])) 
    end
    display(plot!())
end