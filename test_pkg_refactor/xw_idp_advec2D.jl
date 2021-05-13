using Plots
using Revise # reduce need for recompile
using LinearAlgebra

using SparseArrays
using StaticArrays

using UnPack

using NodesAndModes
using NodesAndModes.Tri

using StartUpDG
using StartUpDG.ExplicitTimestepUtils

function build_rhs_matrix(applyRHS,Np,K,vargs...)
    u = zeros(Np,K)
    A = spzeros(Np*K,Np*K)
    for i in eachindex(u)
        u[i] = one(eltype(u))
        r_i = applyRHS(u,vargs...)
        A[:,i] = droptol!(sparse(r_i[:]),1e-12)
        u[i] = zero(eltype(u))
    end
    return A
end

function build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
    # [-1,1,0], [-1,-1,sqrt(4/3)]
    equilateral_map(r,s) = (@. .5*(2*r+1*s+1), @. sqrt(3)*(1+s)/2 - 1/sqrt(3) )
    req,seq = equilateral_map(rq,sq)
    ref,sef = equilateral_map(rf,sf)
    barycentric_coords(r,s) = ((@. (1+r)/2), (@. (1+s)/2), (@. -(r+s)/2))
    λ1,λ2,λ3 = barycentric_coords(rq,sq)
    λ1f,λ2f,λ3f = barycentric_coords(rf,sf)

    Br = diagm(nrJ.*wf)
    Bs = diagm(nsJ.*wf)

    # build extrapolation matrix
    E = zeros(length(rf),length(rq))
    for i = 1:length(rf)
        # d = @. (λ1 - λ1f[i])^2 + (λ2 - λ2f[i])^2 + (λ3 - λ3f[i])^2
        d2 = @. (req-ref[i])^2 + (seq-sef[i])^2
        p = sortperm(d2)
        h2 = (wf[i]/sum(wf))*2/pi # set so that h = radius of circle with area w_i = face weight
        nnbrs = min(4,max(3,count(d2[p] .< h2))) # find 3 closest points
        p = p[1:nnbrs]
        Ei = vandermonde(1,[rf[i]],[sf[i]])/vandermonde(1,rq[p],sq[p])
        E[i,p] = Ei
    end
    E = Matrix(droptol!(sparse(E),1e-13))

    # build stencil
    A = spzeros(Int,length(req),length(req))
    for i = 1:length(req)
        d2 = @. (req-req[i])^2 + (seq-seq[i])^2
        p = sortperm(d2)

        # h^2 = wq[i]/pi = radius of circle with area wq[i]
        # h2 = (sqrt(3)/sum(wq))*wq[i]/pi
        h2 = α^2*(sqrt(3)/sum(wq))*wq[i]/pi

        nnbrs = count(d2[p] .< h2)
        nbrs = p[1:nnbrs]
        A[i,nbrs] .= one(eltype(A))
    end
    A = (A+A')
    A.nzval .= one(eltype(A)) # bool-ish

    scatter(rq,sq,ms=5)
    for ij in eachindex(A)
        if A[ij] != 0
            plot!([rq[ij[1]],rq[ij[2]]],[sq[ij[1]],sq[ij[2]]])
        end
    end
    display(plot!())

    # build graph Laplacian
    L1 = A-diagm(diag(A)) # ignore diag
    L1 -= diagm(vec(sum(L1,dims=2)))

    # Q = S + .5*E'*Br*E, enforce Q*1 = 0
    b1r = -sum(.5*E'*Br*E,dims=2)
    b1s = -sum(.5*E'*Bs*E,dims=2)
    L = [L1 ones(size(L1,2),1);
        ones(1,size(L1,2)) 0]
    br = [b1r;0]
    bs = [b1s;0]
    ψ1r = (L\br)[1:end-1]
    ψ1s = (L\bs)[1:end-1]

    function fillQ(adj,ψ)
        Np = length(ψ)
        S = zeros(Np,Np)
        for i = 1:Np
            for j = 1:Np
                if adj[i,j] != 0
                    S[i,j] += (ψ[j]-ψ[i])
                end
            end
        end
        return S
    end

    S1r,S1s = fillQ.((A,A),(ψ1r,ψ1s))
    Qr = Matrix(droptol!(sparse(S1r + .5*E'*Br*E),1e-14))
    Qs = Matrix(droptol!(sparse(S1s + .5*E'*Bs*E),1e-14))

    return Qr,Qs,E,Br,Bs,A
end

function init_reference_tri_sbp_GQ(N)
    include("SBP_quad_data.jl")
    # initialize a new reference element data struct
    rd = RefElemData()

    fv = tri_face_vertices() # set faces for triangle
    Nfaces = length(fv)
    @pack! rd = fv, Nfaces

    # Construct matrices on reference elements
    r, s = Tri.nodes(N)
    VDM = Tri.vandermonde(N, r, s)
    Vr, Vs = Tri.grad_vandermonde(N, r, s)
    Dr = Vr/VDM
    Ds = Vs/VDM
    @pack! rd = r,s,VDM,Dr,Ds

    # low order interpolation nodes
    r1,s1 = Tri.nodes(1)
    V1 = Tri.vandermonde(1,r,s)/Tri.vandermonde(1,r1,s1)
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

    rq,sq,wq = GQ_SBP[N]
    Vq = Tri.vandermonde(N,rq,sq)/VDM
    M = Vq'*diagm(wq)*Vq
    Pq = M\(Vq'*diagm(wq))
    @pack! rd = rq,sq,wq,Vq,M,Pq

    Vf = Tri.vandermonde(N,rf,sf)/VDM # interpolates from nodes to face nodes
    LIFT = M\(Vf'*diagm(wf)) # lift matrix used in rhs evaluation
    @pack! rd = Vf,LIFT

    # plotting nodes
    rp, sp = Tri.equi_nodes(10)
    Vp = Tri.vandermonde(N,rp,sp)/VDM
    @pack! rd = rp,sp,Vp

    return rd
end

avg(a,b) = .5*(a+b)

"Approximation parameters"
N   = 6 # The order of approximation
K1D = 8
CFL = 0.5
T   = .25 # endtime

# Mesh related variables
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
FToF = connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

# Construct matrices on reference elements
rd = init_reference_tri_sbp_GQ(N)
@unpack r,s,rf,sf,wf,rq,sq,wq,nrJ,nsJ = rd
@unpack VDM,V1,Vq,Vf,Dr,Ds,M,Pq,LIFT = rd

# Need to choose α so that Qr, Qs have zero row sums (and maybe a minimum number of neighbors)
# α = 4 # for N=1
# α = 2.5 #for N=2
α = 3.5 # for N=3
α = 2.5 # for N = 6
Qr,Qs,E,Br,Bs,A = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)
if (norm(sum(Qr,dims=2)) > 1e-10) | (norm(sum(Qs,dims=2)) > 1e-10)
    error("Qr or Qs doesn't sum to zero for α = $α")
end
Qrskew = .5*(Qr-transpose(Qr))
Qsskew = .5*(Qs-transpose(Qs))

# "Construct global coordinates"
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

# "Connectivity maps"
xf,yf = (x->Vf*x).((x,y))
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
mapM = reshape(mapM,length(wf),K)
mapP = reshape(mapP,length(wf),K)

# "Make periodic"
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

# "Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)
nx = nxJ./sJ; ny = nyJ./sJ;

rxJ,sxJ,ryJ,syJ,J = (x->Vq*x).((rxJ,sxJ,ryJ,syJ,J))

M_inv = diagm(@. 1/(wq*J[1][1]))
Pf = transpose(E)*diagm(wf)

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Qrskew,Qsskew,E,Br,Bs,M_inv,Pf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ)
nodemaps = (mapP,mapB)

"initial conditions"
xq = Vq*x
yq = Vq*y
u = @. sin(pi*xq)*sin(pi*yq)
# u = @. exp(-25*(xq^2+yq^2))

"Time integration"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Qrskew,Qsskew,E,Br,Bs, M_inv, Pf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ,wn)
nodemaps = (mapP,mapB)

function rhsx(u,ops,vgeo,fgeo,nodemaps)
    # unpack args
    Qr,Qs,E,Br,Bs,M_inv,EfTW = ops
    rxJ,sxJ,ryJ,syJ,J = vgeo
    nxJ,nyJ,sJ,wn = fgeo
    (mapP,mapB) = nodemaps

    uM = E*u # can replace with nodal extraction
    dudx = rxJ.*(Qr*u) + sxJ.*(Qs*u)
    rhsu = dudx + EfTW*(.5*nxJ.*uM[mapP])
    return rhsu
end

# can check that norm(sum(Qx,dims=2)) ≈ 0 and that Qx = skew-symmetric
Qx = build_rhs_matrix(rhsx,size(u,1),size(u,2),ops,vgeo,fgeo,nodemaps)
invM = spdiagm(0 => 1 ./vec(diagm(wq)*J))

function rhs_global(u,invM,Qx)
    rows = rowvals(Qx)
    vals = nonzeros(Qx)

    rhsu = zero.(u)
    for j = 1:size(Qx,2) # loop over columns
        uj = u[j]
        for index in nzrange(Qx,j) # loop over rows
            i = rows[index]
            Qxij = vals[index]
            dij = abs.(Qxij)
            rhsu[i] += Qxij*uj - dij * (uj-u[i])
        end
    end
    return -invM*rhsu
end

u = vec(u)
for i = 1:Nsteps
    global u # for scoping - these variables are updated

    # Heun's method - this is an example of a 2nd order SSP RK method
    rhsu = rhs_global(u,invM,Qx)
    utmp = u + dt*rhsu
    u .+= .5*dt*(rhsu + rhs_global(utmp,invM,Qx))

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

u = reshape(u,size(xq,1),size(xq,2))

gr(aspect_ratio=1,legend=false,
   markerstrokewidth=0,markersize=2)

# plotting nodes"
rp, sp = equi_nodes(20)
Vp = vandermonde(N,rp,sp)/vandermonde(N,r,s)
vv = Vp*Pq*u
Plots.scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
