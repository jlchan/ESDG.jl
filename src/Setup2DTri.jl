"""
    Module Setup2DTri

Module to aid in setting up 2D meshes, reference operators, geofacs, etc

"""

module Setup2DTri

# non-DG modules
using LinearAlgebra # for diagm
using UnPack # for easy setting/getting in mutable structs

# DG-specific modules
using CommonUtils # for I matrix in geometricFactors
using Basis1D
using Basis2DTri
using UniformTriMesh # for face vertices

export init_reference_tri, init_tri_mesh
export MeshData, RefElemData

mutable struct MeshData

    VX;VY;VZ # vertex coordinates
    K # num elems
    EToV # mesh vertex array
    FToF # face connectivity

    x; y; z # physical points

    mapM; mapP; mapB # connectivity between face nodes

    rxJ; sxJ; txJ; ryJ; syJ; tyJ; rzJ; szJ; tzJ; J # volume geofacs
    nxJ; nyJ; nzJ; sJ # surface geofacs

    MeshData() = new() # empty initializer
end

mutable struct RefElemData

    Nfaces;fv # face vertex tuple list

    r1;s1;t1;V1 # low order interp nodes and matrix

    r; s; t # interpolation nodes
    rq; sq; tq; wq # volume quadrature
    rf; sf; tf; wf # surface quadrature
    rp; sp # plotting nodes

    V # Vandermonde matrix
    Dr; Ds; Dt # differentiation matrices
    Vq; Vf # quadrature interpolation matrix
    Pq # quadrature projection operator
    LIFT # quadrature-based lift matrix
    Vp # plotting interp matrix

    nrJ; nsJ; ntJ # reference normals

    RefElemData() = new() # empty initializer
end

function init_reference_tri(N)

    # initialize a new reference element data struct
    rd = RefElemData()

    fv = tri_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r, s = nodes_2D(N)
    V = vandermonde_2D(N, r, s)
    Vr, Vs = grad_vandermonde_2D(N, r, s)
    Dr = Vr/V
    Ds = Vs/V
    @pack! rd = r,s,V,Dr,Ds

    # low order interpolation nodes
    r1,s1 = nodes_2D(1)
    V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
    @pack! rd = r1,s1,V1

    #Nodes on faces, and face node coordinate
    r1D, w1D = gauss_quad(0,0,N)
    Nfp = length(r1D) # number of points per face
    e = ones(Nfp,1) # vector of all ones
    z = zeros(Nfp,1) # vector of all zeros
    rf = [r1D; -r1D; -e];
    sf = [-e; r1D; -r1D];
    wf = vec(repeat(w1D,3,1));
    nrJ = [z; e; -e]
    nsJ = [-e; e; z]
    @pack! rd = rf,sf,wf,nrJ,nsJ

    Vf = vandermonde_2D(N,rf,sf)/V # interpolates from nodes to face nodes
    invM = (V*V')
    LIFT = invM*(transpose(Vf)*diagm(wf)) # lift matrix used in rhs evaluation
    @pack! rd = Vf,LIFT

    rq,sq,wq = quad_nodes_2D(2*N)
    Vq = vandermonde_2D(N,rq,sq)/V
    @pack! rd = rq,sq,wq,Vq

    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = Pq

    rp, sp = equi_nodes_2D(15)
    Vp = vandermonde_2D(N,rp,sp)/V
    @pack! rd = rp,sp,Vp
    return rd
end

function init_tri_mesh(VXYZ,EToV,rd::RefElemData)

    # initialize a new mesh data struct
    md = MeshData()

    @unpack fv = rd
    VX,VY = VXYZ
    FToF = connect_mesh(EToV,fv)
    Nfaces,K = size(FToF)
    @pack! md = FToF,K,VX,VY,EToV

    #Construct global coordinates
    @unpack r1,s1,V1 = rd
    x = V1*VX[transpose(EToV)]
    y = V1*VY[transpose(EToV)]
    @pack! md = x,y

    #Compute connectivity maps: uP = exterior value used in DG numerical fluxes
    @unpack r,s,Vf = rd
    xf,yf = (x->Vf*x).((x,y))
    mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
    Nfp = convert(Int,size(Vf,1)/Nfaces)
    mapM = reshape(mapM,Nfp*Nfaces,K)
    mapP = reshape(mapP,Nfp*Nfaces,K)
    @pack! md = mapM,mapP,mapB

    #Compute geometric factors and surface normals
    @unpack Dr,Ds = rd
    rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
    @pack! md = rxJ, sxJ, ryJ, syJ, J

    #physical normals are computed via G*nhatJ, G = matrix of geometric terms
    @unpack nrJ,nsJ = rd
    nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
    nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
    sJ = @. sqrt(nxJ^2 + nyJ^2)
    @pack! md = nxJ,nyJ,sJ

    return md
end

end
