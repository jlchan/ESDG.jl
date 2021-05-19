import Triangulate:triangulate

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
function get_boundary_face_labels(triout::TriangulateIO,rd::RefElemData{2,Tri},md::MeshData{2})
    segmentlist = sort(triout.segmentlist,dims=1)
    boundary_faces = findall(vec(md.FToF) .== 1:length(md.FToF))
    boundary_face_tags = zeros(Int,length(boundary_faces))
    for (f,boundary_face) in enumerate(boundary_faces)
        element = (boundary_face - 1) ÷ rd.Nfaces + 1
        face    = (boundary_face - 1) % rd.Nfaces + 1
        vertex_ids = sort(md.EToV[element,rd.fv[face]])
        tag_id = findfirst(c->view(segmentlist,:,c)==vertex_ids,axes(segmentlist,2))
        boundary_face_tags[f] = triout.segmentmarkerlist[tag_id]
    end
    return boundary_face_tags, boundary_faces
end

"""
    function get_node_boundary_tags(triout::TriangulateIO,md::MeshData{2},rd::RefElemData{2,Tri})

Computes node_tags = Nfp x Nfaces * num_elements array where each entry is a Triangulate tag number.
"""
function get_node_boundary_tags(triout::TriangulateIO,rd::RefElemData{2,Tri},md::MeshData{2})
    boundary_face_tags,boundary_faces = get_boundary_face_labels(triout,rd,md)
    node_tags = zeros(Int,size(md.xf,1)÷rd.Nfaces,md.K*rd.Nfaces) # make Nfp x Nfaces*num_elements
    for (i,boundary_face) in enumerate(boundary_faces)
        node_tags[:,boundary_face] .= boundary_face_tags[i]
    end
    node_tags = reshape(node_tags,size(md.xf)...)
end

