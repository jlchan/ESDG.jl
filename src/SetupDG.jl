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

# for basis functions
using NodesAndModes

# using UniformMeshes # for face vertex ordering

# triangular routines
import UniformTriMesh # for face vertices

# quadrilateral routines
import UniformQuadMesh # for face vertices

# hex routines
import UniformHexMesh # for face vertices

# initialization of mesh/reference element data
export init_reference_interval, init_reference_tri
export init_reference_quad, init_reference_hex
export init_mesh
export MeshData, RefElemData

# annotate types for all arrays involved in RHS evaluation
mutable struct RefElemData
    Nfaces; fv # face vertex tuple list

    # non-RHS operators
    V1      # low order interp nodes and matrix
    VDM     # Vandermonde matrix
    Vp      # interp to equispaced nodes

    r; s; t         # interpolation nodes
    rq; sq; tq      # volume quadrature
    rf; sf; tf;     # surface quadrature
    rp; sp; tp      # plotting nodes

    # quadrature weights
    wq::Array{Float64,1}
    wf::Array{Float64,1}

    nrJ; nsJ; ntJ # reference normals

    # differentiation matrices
    Dr::Array{Float64,2}
    Ds::Array{Float64,2}
    Dt::Array{Float64,2}
    Vq::Array{Float64,2}        # quadrature interpolation matrices
    Vf::Array{Float64,2}
    M::Array{Float64,2}         # mass matrix
    Pq::Array{Float64,2}        # L2 projection matrix
    LIFT::Array{Float64,2}      # quadrature-based lift matrix

    RefElemData() = new() # empty initializer
end

# annotate types for geofacs + connectivity arrays for speed in RHS evals
mutable struct MeshData
    VX; VY; VZ              # vertex coordinates
    K::Int                  # num elems
    EToV                    # mesh vertex array
    FToF::Array{Int64,2}    # face connectivity

    x; y; z                 # physical points
    xf; yf; zf
    xq; yq; zq;             # phys quad points, Jacobian-scaled weights
    wJq::Array{Float64,2}

    # arrays of connectivity indices between face nodes
    mapM
    mapP::Array{Int64,2}
    mapB::Array{Int64,1}

    # volume geofacs
    rxJ::Array{Float64,2}
    sxJ::Array{Float64,2}
    txJ::Array{Float64,2}
    ryJ::Array{Float64,2}
    syJ::Array{Float64,2}
    tyJ::Array{Float64,2}
    rzJ::Array{Float64,2}
    szJ::Array{Float64,2}
    tzJ::Array{Float64,2}
    J::Array{Float64,2}

    # surface geofacs
    nxJ::Array{Float64,2}
    nyJ::Array{Float64,2}
    nzJ::Array{Float64,2}
    sJ::Array{Float64,2}

    MeshData() = new() # empty initializer
end

function init_reference_interval(N;Nq=N+1)
    # initialize a new reference element data struct
    rd = RefElemData()

    # Construct matrices on reference elements
    r,_ = gauss_lobatto_quad(0,0,N)
    VDM = vandermonde_1D(N, r)
    Dr = grad_vandermonde_1D(N, r)/VDM
    @pack! rd = r,VDM,Dr

    V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
    @pack! rd = V1

    rq,wq = gauss_quad(0,0,Nq)
    Vq = vandermonde_1D(N, rq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,wq,Vq,M,Pq

    rf = [-1.0;1.0]
    nrJ = [-1.0;1.0]
    Vf = vandermonde_1D(N,rf)/VDM
    LIFT = M\(Vf') # lift matrix
    @pack! rd = rf,nrJ,Vf,LIFT

    # plotting nodes
    rp = LinRange(-1,1,50)
    Vp = vandermonde_1D(N,rp)/VDM
    @pack! rd = rp,Vp

    return rd
end

function init_reference_tri(N;Nq=2*N)
    # initialize a new reference element data struct
    rd = RefElemData()

    fv = UniformTriMesh.tri_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r, s = Tri.nodes_2D(N)
    VDM = Tri.vandermonde_2D(N, r, s)
    Vr, Vs = Tri.grad_vandermonde_2D(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM
    @pack! rd = r,s,VDM,Dr,Ds

    # low order interpolation nodes
    r1,s1 = Tri.nodes_2D(1)
    V1 = Tri.vandermonde_2D(1,r,s)/Tri.vandermonde_2D(1,r1,s1)
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

    rq,sq,wq = Tri.quad_nodes_2D(Nq)
    Vq = Tri.vandermonde_2D(N,rq,sq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,sq,wq,Vq,M,Pq

    Vf = Tri.vandermonde_2D(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(Vf'*diagm(wf)) # lift matrix used in rhs evaluation
    @pack! rd = Vf,LIFT

    # plotting nodes
    rp, sp = Tri.equi_nodes_2D(10)
    Vp = Tri.vandermonde_2D(N,rp,sp)/VDM
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
    r, s = Quad.nodes_2D(N)
    VDM = Quad.vandermonde_2D(N, r, s)
    Vr, Vs = Quad.grad_vandermonde_2D(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM
    @pack! rd = r,s,VDM

    # low order interpolation nodes
    r1,s1 = Quad.nodes_2D(1)
    V1 = Quad.vandermonde_2D(1,r,s)/Quad.vandermonde_2D(1,r1,s1)
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
    # can also use "rq,sq,wq = Quad.quad_nodes_2D(2*N)"
    rq,sq = vec.(meshgrid(r1D))
    wr,ws = vec.(meshgrid(w1D))
    wq = wr .* ws
    Vq = Quad.vandermonde_2D(N,rq,sq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,sq,wq,Vq,M,Pq

    Vf = Quad.vandermonde_2D(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(Vf'*diagm(wf)) # lift matrix used in rhs evaluation

    # expose kronecker product sparsity
    #Dr = droptol!(sparse(Dr), 1e-10)
    #Ds = droptol!(sparse(Ds), 1e-10)
    #Vf = droptol!(sparse(Vf),1e-10)
    #LIFT = droptol!(sparse(LIFT),1e-10)
    @pack! rd = Dr,Ds,Vf,LIFT

    # plotting nodes
    rp, sp = Quad.equi_nodes_2D(15)
    Vp = Quad.vandermonde_2D(N,rp,sp)/VDM
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
    nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ
    nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ
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
    r,s,t = Hex.nodes_3D(N)
    VDM = Hex.vandermonde_3D(N,r,s,t)
    Vr,Vs,Vt = Hex.grad_vandermonde_3D(N,r,s,t)
    Dr,Ds,Dt = (A->A/VDM).(Hex.grad_vandermonde_3D(N,r,s,t))
    @pack! rd = r,s,t,VDM

    # low order interpolation nodes
    r1,s1,t1 = Hex.nodes_3D(1)
    V1 = Hex.vandermonde_3D(1,r,s,t)/Hex.vandermonde_3D(1,r1,s1,t1)
    @pack! rd = V1

    #Nodes on faces, and face node coordinate
    r1D,w1D = quad_nodes_1D
    rquad,squad = vec.(meshgrid(r1D,r1D))
    wr,ws = vec.(meshgrid(w1D,w1D))
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
    rq,sq,tq = vec.(meshgrid(r1D,r1D,r1D))
    wr,ws,wt = vec.(meshgrid(w1D,w1D,w1D))
    wq = wr.*ws.*wt
    Vq = Hex.vandermonde_3D(N,rq,sq,tq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,sq,tq,wq,Vq,M,Pq

    Vf = Hex.vandermonde_3D(N,rf,sf,tf)/VDM
    LIFT = M\(Vf'*diagm(wf))

    # expose kronecker product sparsity
    #Dr = droptol!(sparse(Dr),1e-12)
    #Ds = droptol!(sparse(Ds),1e-12)
    #Dt = droptol!(sparse(Dt),1e-12)
    #Vf = droptol!(sparse(Vf),1e-12)
    #LIFT = droptol!(sparse(LIFT),1e-12)
    @pack! rd = Dr,Ds,Dt,Vf,LIFT

    # plotting nodes
    rp,sp,tp = Hex.equi_nodes_3D(15)
    Vp = Hex.vandermonde_3D(N,rp,sp,tp)/VDM
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
