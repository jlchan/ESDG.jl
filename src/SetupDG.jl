"""
    Module Setup2DTri

Module to aid in setting up 2D meshes, reference operators, geofacs, etc

"""

module SetupDG

# non-DG modules
using LinearAlgebra # for diagm
using SparseArrays # for sparse, droptol
using UnPack # for easy setting/getting in mutable structs

# DG-specific modules
using CommonUtils # for I matrix in geometricFactors
using Basis1D

# triangular routines
import Basis2DTri
import UniformTriMesh # for face vertices

# quadrilateral routines
import Basis2DQuad
import UniformQuadMesh # for face vertices

export init_reference_tri, init_reference_quad
export init_mesh_2D
export MeshData, RefElemData

mutable struct MeshData

    VX;VY;VZ # vertex coordinates
    K # num elems
    EToV # mesh vertex array
    FToF # face connectivity

    x; y; z # physical points
    xf;yf;zf
    xq;yq;zq;wJq # phys quad points, Jacobian-scaled weights

    mapM; mapP; mapB # connectivity between face nodes

    rxJ; sxJ; txJ; ryJ; syJ; tyJ; rzJ; szJ; tzJ; J # volume geofacs
    nxJ; nyJ; nzJ; sJ # surface geofacs

    MeshData() = new() # empty initializer
end

mutable struct RefElemData

    Nfaces; fv # face vertex tuple list

    V1 # low order interp nodes and matrix

    # probably won't use nodes, but might as well keep them around
    r; s; t # interpolation nodes
    rq; sq; tq; wq # volume quadrature
    rf; sf; tf; wf # surface quadrature
    rp; sp; tp # plotting nodes

    VDM # Vandermonde matrix
    Dr; Ds; Dt # differentiation matrices
    Vq; Vf # quadrature interpolation matrices
    M; Pq # mass matrix, L2 projection matrix
    LIFT # quadrature-based lift matrix
    Vp # interp to equispaced nodes

    nrJ; nsJ; ntJ # reference normals

    RefElemData() = new() # empty initializer
end

function init_reference_tri(N)

    # initialize a new reference element data struct
    rd = RefElemData()

    fv = UniformTriMesh.tri_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r, s = Basis2DTri.nodes_2D(N)
    VDM = Basis2DTri.vandermonde_2D(N, r, s)
    Vr, Vs = Basis2DTri.grad_vandermonde_2D(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM
    @pack! rd = r,s,VDM,Dr,Ds

    # low order interpolation nodes
    r1,s1 = Basis2DTri.nodes_2D(1)
    V1 = Basis2DTri.vandermonde_2D(1,r,s)/Basis2DTri.vandermonde_2D(1,r1,s1)
    @pack! rd = V1

    #Nodes on faces, and face node coordinate
    r1D, w1D = gauss_quad(0,0,N)
    Nfp = length(r1D) # number of points per face
    e = ones(Nfp) # vector of all ones
    z = zeros(Nfp) # vector of all zeros
    rf = [r1D; -r1D; -e];
    sf = [-e; r1D; -r1D];
    wf = vec(repeat(w1D,3,1));
    nrJ = [z; e; -e]
    nsJ = [-e; e; z]
    @pack! rd = rf,sf,wf,nrJ,nsJ

    rq,sq,wq = Basis2DTri.quad_nodes_2D(2*N)
    Vq = Basis2DTri.vandermonde_2D(N,rq,sq)/VDM
    M = transpose(Vq)*diagm(wq)*Vq
    Pq = M\(transpose(Vq)*diagm(wq))
    @pack! rd = rq,sq,wq,Vq,M,Pq

    Vf = Basis2DTri.vandermonde_2D(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix used in rhs evaluation
    @pack! rd = Vf,LIFT

    # plotting nodes
    rp, sp = Basis2DTri.equi_nodes_2D(15)
    Vp = Basis2DTri.vandermonde_2D(N,rp,sp)/VDM
    @pack! rd = rp,sp,Vp

    return rd
end

# default to full quadrature nodes
# if quad_nodes_1D=tuple of (r1D,w1D) is supplied, use those nodes
function init_reference_quad(N,quad_nodes_1D = gauss_quad(0,0,N))

    # initialize a new reference element data struct
    rd = RefElemData()

    fv = UniformQuadMesh.quad_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r, s = Basis2DQuad.nodes_2D(N)
    VDM = Basis2DQuad.vandermonde_2D(N, r, s)
    Vr, Vs = Basis2DQuad.grad_vandermonde_2D(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM
    @pack! rd = r,s,VDM

    # low order interpolation nodes
    r1,s1 = Basis2DQuad.nodes_2D(1)
    V1 = Basis2DQuad.vandermonde_2D(1,r,s)/Basis2DQuad.vandermonde_2D(1,r1,s1)
    @pack! rd = V1

    #Nodes on faces, and face node coordinate
    # r1D,w1D = quad_nodes_1D(0,0,N)
    # r1D,w1D = gauss_lobatto_quad(0,0,N)
    # r1D,w1D = gauss_quad(0,0,N)
    r1D,w1D = quad_nodes_1D
    Nfp = length(r1D)
    e = ones(size(r1D))
    z = zeros(size(r1D))
    rf = [r1D; e; -r1D; -e]
    sf = [-e; r1D; e; -r1D]
    wf = vec(repeat(w1D,Nfaces,1));
    nrJ = [z; e; z; -e]
    nsJ = [-e; z; e; z]
    @pack! rd = rf,sf,wf,nrJ,nsJ

    # quadrature nodes - build from 1D nodes.
    # can also use "rq,sq,wq = Basis2DQuad.quad_nodes_2D(2*N)"
    rq,sq = (x->x[:]).(meshgrid(r1D))
    wr,ws = meshgrid(w1D)
    wq = wr[:] .* ws[:]
    Vq = Basis2DQuad.vandermonde_2D(N,rq,sq)/VDM
    M = transpose(Vq)*diagm(wq)*Vq
    Pq = M\(transpose(Vq)*diagm(wq))
    @pack! rd = rq,sq,wq,Vq,M,Pq

    Vf = Basis2DQuad.vandermonde_2D(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix used in rhs evaluation

    # expose kronecker product sparsity
    Dr = droptol!(sparse(Dr), 1e-10)
    Ds = droptol!(sparse(Ds), 1e-10)
    Vf = droptol!(sparse(Vf),1e-10)
    LIFT = droptol!(sparse(LIFT),1e-10)
    @pack! rd = Dr,Ds,Vf,LIFT

    # plotting nodes
    rp, sp = Basis2DQuad.equi_nodes_2D(15)
    Vp = Basis2DQuad.vandermonde_2D(N,rp,sp)/VDM
    @pack! rd = rp,sp,Vp

    return rd
end

function init_mesh_2D(VXYZ,EToV,rd::RefElemData)

    # initialize a new mesh data struct
    md = MeshData()

    @unpack fv = rd
    VX,VY = VXYZ
    FToF = connect_mesh(EToV,fv)
    Nfaces,K = size(FToF)
    @pack! md = FToF,K,VX,VY,EToV

    #Construct global coordinates
    @unpack V1 = rd
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
    @pack! md = xf,yf,mapM,mapP,mapB

    #Compute geometric factors and surface normals
    @unpack Dr,Ds = rd
    rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
    @pack! md = rxJ, sxJ, ryJ, syJ, J

    @unpack Vq,wq = rd
    xq,yq = (x->Vq*x).((x,y))
    wJq = diagm(wq)*(Vq*J)
    @pack! md = xq,yq,wJq

    #physical normals are computed via G*nhatJ, G = matrix of geometric terms
    @unpack nrJ,nsJ = rd
    nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
    nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
    sJ = @. sqrt(nxJ^2 + nyJ^2)
    @pack! md = nxJ,nyJ,sJ

    return md
end

end
