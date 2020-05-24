"""
Module Setup2D

Setup codes for reference element operators, meshes, geometric terms.

"""

module SetupDG

using LinearAlgebra # for diagm
using SparseArrays # for sparse, droptol
using UnPack # for easy setting/getting in mutable structs

using CommonUtils # for I matrix in geometricFactors

# 1D routines
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
export init_reference_interval
export init_reference_tri
export init_reference_quad
export init_reference_hex
export init_mesh
export MeshData, RefElemData

# mutable in case we need to modify elements (e.g., interpolate to quadrature)
Base.@kwdef mutable struct RefElemData{Tv}

    Nfaces::Union{Int,Missing} = missing
    fv::Union{Array{Array{Int,1},1},Missing} = missing # face vertex tuple list - only needed for 1D and up

    # interpolation nodes - specify at least r (for 1D)
    r::Array{Tv,1}
    s::Union{Array{Tv,1},Missing} = missing
    t::Union{Array{Tv,1},Missing} = missing

    # volume quadrature nodes/weights
    rq::Array{Tv,1}
    sq::Union{Array{Tv,1},Missing} = missing
    tq::Union{Array{Tv,1},Missing} = missing
    wq::Array{Tv,1}

    # surface quadrature nodes/weights
    rf::Array{Tv,1}
    sf::Union{Array{Tv,1},Missing} = missing
    tf::Union{Array{Tv,1},Missing} = missing
    wf::Union{Array{Tv,1},Missing} = missing

    # plotting nodes
    rp::AbstractArray{Tv,1} # includes ranges
    sp::Union{AbstractArray{Tv,1},Missing} = missing
    tp::Union{AbstractArray{Tv,1},Missing} = missing

    # reference scaled normals
    nrJ::Array{Tv,1}
    nsJ::Union{Array{Tv,1},Missing} = missing
    ntJ::Union{Array{Tv,1},Missing} = missing

    # interpolation and setup matrices
    VDM::Array{Tv,2}   # Vandermonde matrix (implicitly defines Lagrange bases)
    V1::Array{Tv,2}    # (bi/tri)-linear interpolation matrix to high order nodes
    Vp::Array{Tv,2}    # interpolation matrix to equispaced nodes

    # let operators be abstract arrays (dense or sparse)
    # differentiation matrices
    Dr::AbstractArray{Tv,2}
    Ds::Union{AbstractArray{Tv,2},Missing} = missing
    Dt::Union{AbstractArray{Tv,2},Missing} = missing

    # quadrature based matrices
    Vq::AbstractArray{Tv,2}    # interpolate to volume quad nodes
    Vf::AbstractArray{Tv,2}    # interpolate to surface quad nodes
    M::AbstractArray{Tv,2}     # mass matrix
    Pq::AbstractArray{Tv,2}    # quadrature L2 projection matrix
    LIFT::AbstractArray{Tv,2}  # quadrature surface lift matrix

end

Base.@kwdef mutable struct MeshData{Tv}

    # vertex coordinates and mesh connectivity arrays
    VX::Array{Tv,1}
    VY::Union{Array{Tv,1},Missing} = missing
    VZ::Union{Array{Tv,1},Missing} = missing
    EToV::Array{Int,2} # mesh vertex array
    FToF::Array{Int,2} # face connectivity
    K::Int # num elems

    # physical points
    x::Array{Tv,2}
    y::Union{Array{Tv,2},Missing} = missing
    z::Union{Array{Tv,2},Missing} = missing

    # face points
    xf::Array{Tv,2}
    yf::Union{Array{Tv,2},Missing} = missing
    zf::Union{Array{Tv,2},Missing} = missing

    # phys quad points, Jacobian-scaled weights
    xq::Array{Tv,2}
    yq::Union{Array{Tv,2},Missing} = missing
    zq::Union{Array{Tv,2},Missing} = missing
    wJq::Array{Tv,2}

    # connectivity maps between face nodes
    mapM::Array{Int,2}
    mapP::Array{Int,2}
    mapB::Array # either Array{1,2}

    # volume geofacs
    rxJ::Array{Tv,2}
    sxJ::Union{Array{Tv,2},Missing} = missing
    txJ::Union{Array{Tv,2},Missing} = missing
    ryJ::Union{Array{Tv,2},Missing} = missing
    syJ::Union{Array{Tv,2},Missing} = missing
    tyJ::Union{Array{Tv,2},Missing} = missing
    rzJ::Union{Array{Tv,2},Missing} = missing
    szJ::Union{Array{Tv,2},Missing} = missing
    tzJ::Union{Array{Tv,2},Missing} = missing
    J::Array{Tv,2}

    # surface geofacs
    nxJ::Array{Tv,2}
    nyJ::Union{Array{Tv,2},Missing} = missing
    nzJ::Union{Array{Tv,2},Missing} = missing
    sJ::Array{Tv,2}

    # MeshData() = new() # empty initializer
end

function init_reference_interval(N)

    # Construct matrices on reference elements
    r,_ = gauss_lobatto_quad(0,0,N)
    VDM = vandermonde_1D(N, r)
    Dr = grad_vandermonde_1D(N, r)/VDM

    V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])

    rq,wq = gauss_quad(0,0,N+1)
    Vq = vandermonde_1D(N, rq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))

    rf = [-1.0;1.0]
    nrJ = [-1.0;1.0]
    Vf = vandermonde_1D(N,rf)/VDM
    LIFT = M\(transpose(Vf)) # lift matrix

    # plotting nodes
    rp = LinRange(-1,1,50)
    Vp = vandermonde_1D(N,rp)/VDM

    # return rd
    @show typeof.((r,VDM,V1,rq,wq,Vq,Dr,M,Pq,rf,nrJ,Vf,LIFT,rp,Vp))
    return RefElemData(r=r,VDM=VDM,V1=V1,
                        rq=rq,wq=wq,Vq=Vq,
                        Dr=Dr,M=M,Pq=Pq,
                        rf=rf,nrJ=nrJ,
                        Vf=Vf,LIFT=LIFT,
                        rp=rp,Vp=Vp)

end

function init_reference_tri(N)

    fv = UniformTriMesh.tri_face_vertices() # set faces for triangle
    Nfaces = length(fv)

    # Construct matrices on reference elements
    r, s = Basis2DTri.nodes_2D(N)
    VDM = Basis2DTri.vandermonde_2D(N, r, s)
    Vr, Vs = Basis2DTri.grad_vandermonde_2D(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM

    # low order interpolation nodes
    r1,s1 = Basis2DTri.nodes_2D(1)
    V1 = Basis2DTri.vandermonde_2D(1,r,s)/Basis2DTri.vandermonde_2D(1,r1,s1)

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

    rq,sq,wq = Basis2DTri.quad_nodes_2D(2*N)
    Vq = Basis2DTri.vandermonde_2D(N,rq,sq)/VDM
    M = transpose(Vq)*diagm(wq)*Vq
    Pq = M\(transpose(Vq)*diagm(wq))

    Vf = Basis2DTri.vandermonde_2D(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix used in rhs evaluation

    # plotting nodes
    rp, sp = Basis2DTri.equi_nodes_2D(15)
    Vp = Basis2DTri.vandermonde_2D(N,rp,sp)/VDM

    return RefElemData(Nfaces=Nfaces,fv=fv,
                    r=r,s=s,
                    VDM=VDM,V1=V1,
                    Dr=Dr,Ds=Ds,
                    rq=rq,sq=sq,wq=wq,
                    Vq=Vq,M=M,Pq=Pq,
                    rf=rf,sf=sf,wf=wf,
                    nrJ=nrJ,nsJ=nsJ,
                    Vf=Vf,LIFT=LIFT,
                    rp=rp,sp=sp,Vp=Vp)
end

"
init_reference_quad(N,quad_nodes_1D = gauss_quad(0,0,N))
 if quad_nodes_1D=tuple of (r1D,w1D) is supplied, uses those nodes instead
"
function init_reference_quad(N,quad_nodes_1D = gauss_quad(0,0,N))

    fv = UniformQuadMesh.quad_face_vertices() # set faces for triangle
    Nfaces = length(fv)

    # Construct matrices on reference elements
    r, s = Basis2DQuad.nodes_2D(N)
    VDM = Basis2DQuad.vandermonde_2D(N, r, s)
    Vr, Vs = Basis2DQuad.grad_vandermonde_2D(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM

    # low order interpolation nodes
    r1,s1 = Basis2DQuad.nodes_2D(1)
    V1 = Basis2DQuad.vandermonde_2D(1,r,s)/Basis2DQuad.vandermonde_2D(1,r1,s1)

    #Nodes on faces, and face node coordinate
    r1D,w1D = quad_nodes_1D
    Nfp = length(r1D)
    e = ones(size(r1D))
    z = zeros(size(r1D))
    rf = [r1D; e; -r1D; -e]
    sf = [-e; r1D; e; -r1D]
    wf = vec(repeat(w1D,Nfaces,1));
    nrJ = [z; e; z; -e]
    nsJ = [-e; z; e; z]

    # quadrature nodes - build from 1D nodes.
    # can also use "rq,sq,wq = Basis2DQuad.quad_nodes_2D(2*N)"
    rq,sq = (x->x[:]).(meshgrid(r1D))
    wr,ws = meshgrid(w1D)
    wq = wr[:] .* ws[:]
    Vq = Basis2DQuad.vandermonde_2D(N,rq,sq)/VDM
    M = transpose(Vq)*diagm(wq)*Vq
    Pq = M\(transpose(Vq)*diagm(wq))

    Vf = Basis2DQuad.vandermonde_2D(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix used in rhs evaluation

    # expose kronecker product sparsity
    Dr = droptol!(sparse(Dr), 1e-10)
    Ds = droptol!(sparse(Ds), 1e-10)
    Vf = droptol!(sparse(Vf),1e-10)
    LIFT = droptol!(sparse(LIFT),1e-10)

    # plotting nodes
    rp, sp = Basis2DQuad.equi_nodes_2D(15)
    Vp = Basis2DQuad.vandermonde_2D(N,rp,sp)/VDM

    # return rd
    return RefElemData(Nfaces=Nfaces,fv=fv,
                    r=r,s=s,
                    VDM=VDM,V1=V1,
                    Dr=Dr,Ds=Ds,
                    rq=rq,sq=sq,wq=wq,
                    Vq=Vq,M=M,Pq=Pq,
                    rf=rf,sf=sf,wf=wf,
                    nrJ=nrJ,nsJ=nsJ,
                    Vf=Vf,LIFT=LIFT,
                    rp=rp,sp=sp,Vp=Vp)
end

# dispatch to 2D or 3D version if tuple called
function init_mesh(VXYZ,EToV,rd::RefElemData)
    return init_mesh(VXYZ...,EToV,rd)
end

function init_mesh(VX,VY,EToV,rd::RefElemData)

    # # initialize a new mesh data struct
    # md = MeshData()

    @unpack fv = rd
    FToF = connect_mesh(EToV,fv)
    Nfaces,K = size(FToF)
    # @pack! md = FToF,K,VX,VY,EToV

    #Construct global coordinates
    @unpack V1 = rd
    x = V1*VX[transpose(EToV)]
    y = V1*VY[transpose(EToV)]
    # @pack! md = x,y

    #Compute connectivity maps: uP = exterior value used in DG numerical fluxes
    @unpack r,s,Vf = rd
    xf,yf = (x->Vf*x).((x,y))
    mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
    Nfp = convert(Int,size(Vf,1)/Nfaces)
    mapM = reshape(mapM,Nfp*Nfaces,K)
    mapP = reshape(mapP,Nfp*Nfaces,K)
    # @pack! md = xf,yf,mapM,mapP,mapB

    #Compute geometric factors and surface normals
    @unpack Dr,Ds = rd
    rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
    # @pack! md = rxJ, sxJ, ryJ, syJ, J

    @unpack Vq,wq = rd
    xq,yq = (x->Vq*x).((x,y))
    wJq = diagm(wq)*(Vq*J)
    # @pack! md = xq,yq,wJq

    #physical normals are computed via G*nhatJ, G = matrix of geometric terms
    @unpack nrJ,nsJ = rd
    nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ
    nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ
    sJ = @. sqrt(nxJ^2 + nyJ^2)
    # @pack! md = nxJ,nyJ,sJ

    # return md
    return MeshData(K=K,VX=VX,VY=VY,
                EToV=EToV,FToF=FToF,
                x=x,y=y,xf=xf,yf=yf,
                mapM=mapM,mapP=mapP,mapB=mapB,
                rxJ=rxJ,sxJ=sxJ,ryJ=ryJ,syJ=syJ,J=J,
                nxJ=nxJ,nyJ=nyJ,sJ=sJ,
                xq=xq,yq=yq,wJq=wJq)
end


"========== 3D routines ============="

function init_reference_hex(N,quad_nodes_1D=gauss_quad(0,0,N))

    # # initialize a new reference element data struct
    # rd = RefElemData()

    fv = UniformHexMesh.hex_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    # @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r,s,t = Basis3DHex.nodes_3D(N)
    VDM = Basis3DHex.vandermonde_3D(N,r,s,t)
    Vr,Vs,Vt = Basis3DHex.grad_vandermonde_3D(N,r,s,t)
    Dr,Ds,Dt = (A->A/VDM).(Basis3DHex.grad_vandermonde_3D(N,r,s,t))
    # @pack! rd = r,s,t,VDM

    # low order interpolation nodes
    r1,s1,t1 = Basis3DHex.nodes_3D(1)
    V1 = Basis3DHex.vandermonde_3D(1,r,s,t)/Basis3DHex.vandermonde_3D(1,r1,s1,t1)
    # @pack! rd = V1

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

    # @pack! rd = rf,sf,tf,wf,nrJ,nsJ,ntJ

    # quadrature nodes - build from 1D nodes.
    rq,sq,tq = vec.(meshgrid(r1D,r1D,r1D))
    wr,ws,wt = vec.(meshgrid(w1D,w1D,w1D))
    wq = @. wr*ws*wt
    Vq = Basis3DHex.vandermonde_3D(N,rq,sq,tq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    # @pack! rd = rq,sq,tq,wq,Vq,M,Pq

    Vf = Basis3DHex.vandermonde_3D(N,rf,sf,tf)/VDM
    LIFT = M\(Vf'*diagm(wf))

    # expose kronecker product sparsity
    Dr = droptol!(sparse(Dr),1e-12)
    Ds = droptol!(sparse(Ds),1e-12)
    Dt = droptol!(sparse(Dt),1e-12)
    Vf = droptol!(sparse(Vf),1e-12)
    LIFT = droptol!(sparse(LIFT),1e-12)
    # @pack! rd = Dr,Ds,Dt,Vf,LIFT

    # plotting nodes
    rp,sp,tp = Basis3DHex.equi_nodes_3D(15)
    Vp = Basis3DHex.vandermonde_3D(N,rp,sp,tp)/VDM
    # @pack! rd = rp,sp,tp,Vp

    # return rd
    return RefElemData(Nfaces=Nfaces,fv=fv,
                    r=r,s=s,t=t,
                    VDM=VDM,V1=V1,
                    Dr=Dr,Ds=Ds,Dt=Dt,
                    rq=rq,sq=sq,tq=tq,wq=wq,
                    Vq=Vq,M=M,Pq=Pq,
                    rf=rf,sf=sf,tf=tf,wf=wf,
                    nrJ=nrJ,nsJ=nsJ,ntJ=ntJ,
                    Vf=Vf,LIFT=LIFT,
                    rp=rp,sp=sp,tp=tp,Vp=Vp)
end

function init_mesh(VX,VY,VZ,EToV,rd::RefElemData)

    # # initialize a new mesh data struct
    # md = MeshData()

    @unpack fv = rd
    FToF = connect_mesh(EToV,fv)
    Nfaces,K = size(FToF)
    # @pack! md = FToF,K,VX,VY,VZ,EToV

    #Construct global coordinates
    @unpack V1 = rd
    x = V1*VX[transpose(EToV)]
    y = V1*VY[transpose(EToV)]
    z = V1*VZ[transpose(EToV)]
    # @pack! md = x,y,z

    #Compute connectivity maps: uP = exterior value used in DG numerical fluxes
    @unpack r,s,t,Vf = rd
    xf,yf,zf = (x->Vf*x).((x,y,z))
    mapM,mapP,mapB = build_node_maps((xf,yf,zf),FToF)
    Nfp = convert(Int,size(Vf,1)/Nfaces)
    mapM = reshape(mapM,Nfp*Nfaces,K)
    mapP = reshape(mapP,Nfp*Nfaces,K)
    # @pack! md = xf,yf,zf,mapM,mapP,mapB

    #Compute geometric factors and surface normals
    @unpack Dr,Ds,Dt = rd
    rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J = geometric_factors(x,y,z,Dr,Ds,Dt)
    # @pack! md = rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J

    @unpack Vq,wq = rd
    xq,yq,zq = (x->Vq*x).((x,y,z))
    wJq = diagm(wq)*(Vq*J)
    # @pack! md = xq,yq,zq,wJq

    #physical normals are computed via G*nhatJ, G = matrix of geometric terms
    @unpack nrJ,nsJ,ntJ = rd
    nxJ = nrJ.*(Vf*rxJ) + nsJ.*(Vf*sxJ) + ntJ.*(Vf*txJ)
    nyJ = nrJ.*(Vf*ryJ) + nsJ.*(Vf*syJ) + ntJ.*(Vf*tyJ)
    nzJ = nrJ.*(Vf*rzJ) + nsJ.*(Vf*szJ) + ntJ.*(Vf*tzJ)
    sJ = @. sqrt(nxJ^2 + nyJ^2 + nzJ^2)
    # @pack! md = nxJ,nyJ,nzJ,sJ

    # return md
    return MeshData(K=K,VX=VX,VY=VY,VZ=VZ,
                EToV=EToV,FToF=FToF,
                x=x,y=y,z=z,xf=xf,yf=yf,zf=zf,
                mapM=mapM,mapP=mapP,mapB=mapB,
                rxJ=rxJ,sxJ=sxJ,txJ=txJ,
                ryJ=ryJ,syJ=syJ,tyJ=tyJ,
                rzJ=rzJ,szJ=szJ,tzJ=tzJ,J=J,
                nxJ=nxJ,nyJ=nyJ,nzJ=nzJ,sJ=sJ,
                xq=xq,yq=yq,zq=zq,wJq=wJq)
end

end
