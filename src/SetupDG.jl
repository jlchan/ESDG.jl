"""
Module Setup2D

Module to aid in setting up reference operators, meshes, geometric terms
Also provides functions to assemble global DG-SBP operators

"""

module SetupDG

# non-DG modules
using LinearAlgebra # for diagm
using SparseArrays # for sparse, droptol
using UnPack # for easy setting/getting in mutable structs

# matlab-like modules
using CommonUtils # for I matrix in geometricFactors
using Basis1D

# triangular routines
import Basis2DTri
import UniformTriMesh # for face vertices

# quadrilateral routines
import Basis2DQuad
import UniformQuadMesh # for face vertices

# hex routines
import Basis3DHex
import UniformHexMesh # for face vertices

# initialization of mesh/reference element data
export init_reference_interval, init_reference_tri
export init_reference_quad, init_reference_hex
export init_mesh
export MeshData, RefElemData

# do we realy need mutable here?  I think we can remove it
# TODO: add type annotation to all of these, switch to immutable struct
mutable struct MeshData

    VX;VY;VZ # vertex coordinates
    K::Int # num elems
    EToV # mesh vertex array
    FToF # face connectivity

    x; y; z # physical points
    xf;yf;zf
    xq;yq;zq;wJq # phys quad points, Jacobian-scaled weights

    mapM; mapP; mapB # connectivity between face nodes

    # volume geofacs
    rxJ; sxJ; txJ
    ryJ; syJ; tyJ
    rzJ; szJ; tzJ; J

    # surface geofacs
    nxJ; nyJ; nzJ; sJ

    MeshData() = new() # empty initializer


end

mutable struct RefElemData

    Nfaces; fv # face vertex tuple list

    V1 # low order interp nodes and matrix

    # probably won't use nodes, but might as well keep them around
    r; s; t        # interpolation nodes
    rq; sq; tq; wq # volume quadrature
    rf; sf; tf; wf # surface quadrature
    rp; sp; tp     # plotting nodes

    VDM         # Vandermonde matrix
    Dr; Ds; Dt  # differentiation matrices
    Vq; Vf      # quadrature interpolation matrices
    M; Pq       # mass matrix, L2 projection matrix
    LIFT        # quadrature-based lift matrix
    Vp          # interp to equispaced nodes

    nrJ; nsJ; ntJ # reference normals

    RefElemData() = new() # empty initializer
end

function init_reference_interval(N)
    # initialize a new reference element data struct
    rd = RefElemData()

    # Construct matrices on reference elements
    r,_ = gauss_lobatto_quad(0,0,N)
    VDM = vandermonde_1D(N, r)
    Dr = grad_vandermonde_1D(N, r)/VDM
    @pack! rd = r,VDM,Dr

    V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
    @pack! rd = V1

    rq,wq = gauss_quad(0,0,N+1)
    Vq = vandermonde_1D(N, rq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,wq,Vq,M,Pq

    rf = [-1;1]
    nrJ = [-1;1]
    Vf = vandermonde_1D(N,rf)/VDM
    LIFT = M\(transpose(Vf)) # lift matrix
    @pack! rd = rf,nrJ,Vf,LIFT

    # plotting nodes
    rp = LinRange(-1,1,50)
    Vp = vandermonde_1D(N,rp)/VDM
    @pack! rd = rp,Vp

    return rd

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

# dispatch to 2D or 3D version if tuple called
function init_mesh(VXYZ,EToV,rd::RefElemData)
    return init_mesh(VXYZ...,EToV,rd)
end

function init_mesh(VX,VY,EToV,rd::RefElemData)

    # initialize a new mesh data struct
    md = MeshData()

    @unpack fv = rd
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


"========== 3D routines ============="

function init_reference_hex(N,quad_nodes_1D=gauss_quad(0,0,N))

    # initialize a new reference element data struct
    rd = RefElemData()

    fv = UniformHexMesh.hex_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r,s,t = Basis3DHex.nodes_3D(N)
    VDM = Basis3DHex.vandermonde_3D(N,r,s,t)
    Vr,Vs,Vt = Basis3DHex.grad_vandermonde_3D(N,r,s,t)
    Dr,Ds,Dt = (A->A/VDM).(Basis3DHex.grad_vandermonde_3D(N,r,s,t))
    @pack! rd = r,s,t,VDM

    # low order interpolation nodes
    r1,s1,t1 = Basis3DHex.nodes_3D(1)
    V1 = Basis3DHex.vandermonde_3D(1,r,s,t)/Basis3DHex.vandermonde_3D(1,r1,s1,t1)
    @pack! rd = V1

    #Nodes on faces, and face node coordinate
    r1D,w1D = quad_nodes_1D
    rquad,squad = (x->x[:]).(meshgrid(r1D,r1D))
    wr,ws = (x->x[:]).(meshgrid(w1D,w1D))
    wquad = wr.*ws
    e = ones(size(rquad))
    zz = zeros(size(rquad))
    rf = [-e; e; rquad; rquad; rquad; rquad]
    sf = [rquad; rquad; -e; e; squad; squad]
    tf = [squad; squad; squad; squad; -e; e]
    wf = vec(repeat(wquad,Nfaces,1));
    nrJ = [-e; e; zz;zz; zz;zz]
    nsJ = [zz;zz; -e; e; zz;zz]
    ntJ = [zz;zz; zz;zz; -e; e]

    @pack! rd = rf,sf,tf,wf,nrJ,nsJ,ntJ

    # quadrature nodes - build from 1D nodes.
    rq,sq,tq = meshgrid(r1D,r1D,r1D)
    wr,ws,wt = meshgrid(w1D,w1D,w1D)
    wq = wr.*ws.*wt
    Vq = Basis3DHex.vandermonde_3D(N,rq,sq,tq)/VDM
    M = transpose(Vq)*diagm(wq)*Vq
    Pq = M\(transpose(Vq)*diagm(wq))
    @pack! rd = rq,sq,tq,wq,Vq,M,Pq

    Vf = Basis3DHex.vandermonde_3D(N,rf,sf,tf)/VDM
    LIFT = M\(transpose(Vf)*diagm(wf))

    # expose kronecker product sparsity
    Dr = droptol!(sparse(Dr),1e-12)
    Ds = droptol!(sparse(Ds),1e-12)
    Dt = droptol!(sparse(Dt),1e-12)
    Vf = droptol!(sparse(Vf),1e-12)
    LIFT = droptol!(sparse(LIFT),1e-12)
    @pack! rd = Dr,Ds,Dt,Vf,LIFT

    # plotting nodes
    rp,sp,tp = Basis3DHex.equi_nodes_3D(15)
    Vp = Basis3DHex.vandermonde_3D(N,rp,sp,tp)/VDM
    @pack! rd = rp,sp,tp,Vp

    return rd
end

function init_mesh(VX,VY,VZ,EToV,rd::RefElemData)

    # initialize a new mesh data struct
    md = MeshData()

    @unpack fv = rd
    FToF = connect_mesh(EToV,fv)
    Nfaces,K = size(FToF)
    @pack! md = FToF,K,VX,VY,VZ,EToV

    #Construct global coordinates
    @unpack V1 = rd
    x = V1*VX[transpose(EToV)]
    y = V1*VY[transpose(EToV)]
    z = V1*VZ[transpose(EToV)]
    @pack! md = x,y,z

    #Compute connectivity maps: uP = exterior value used in DG numerical fluxes
    @unpack r,s,t,Vf = rd
    xf,yf,zf = (x->Vf*x).((x,y,z))
    mapM,mapP,mapB = build_node_maps((xf,yf,zf),FToF)
    Nfp = convert(Int,size(Vf,1)/Nfaces)
    mapM = reshape(mapM,Nfp*Nfaces,K)
    mapP = reshape(mapP,Nfp*Nfaces,K)
    @pack! md = xf,yf,zf,mapM,mapP,mapB

    #Compute geometric factors and surface normals
    @unpack Dr,Ds,Dt = rd
    rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J = geometric_factors(x,y,z,Dr,Ds,Dt)
    @pack! md = rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J

    @unpack Vq,wq = rd
    xq,yq,zq = (x->Vq*x).((x,y,z))
    wJq = diagm(wq)*(Vq*J)
    @pack! md = xq,yq,zq,wJq

    #physical normals are computed via G*nhatJ, G = matrix of geometric terms
    @unpack nrJ,nsJ,ntJ = rd
    nxJ = nrJ.*(Vf*rxJ) + nsJ.*(Vf*sxJ) + ntJ.*(Vf*txJ)
    nyJ = nrJ.*(Vf*ryJ) + nsJ.*(Vf*syJ) + ntJ.*(Vf*tyJ)
    nzJ = nrJ.*(Vf*rzJ) + nsJ.*(Vf*szJ) + ntJ.*(Vf*tzJ)
    sJ = @. sqrt(nxJ.^2 + nyJ.^2 + nzJ.^2)
    @pack! md = nxJ,nyJ,nzJ,sJ

    return md
end

end
