using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./src")
using CommonUtils
using NodesAndModes
using NodesAndModes.Tri

using UniformTriMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

"Approximation parameters"
N = 10 # The order of approximation
K1D = 4
CFL = 1.0 # CFL goes up to 2.5ish
T = 2/3 #1.0 # endtime

"Mesh related variables"
VX, VY, EToV = uniform_tri_mesh(K1D)

# initialize ref element and mesh
rd = init_reference_tri(N)
md = init_mesh((VX,VY),EToV,rd)

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

## construct hybridized SBP operators

@unpack M,Dr,Ds,Vq,Pq,Vf,wf,nrJ,nsJ = rd
Qr = Pq'*M*Dr*Pq
Qs = Pq'*M*Ds*Pq
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Bs = diagm(wf.*nsJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
         -Br*Ef  Br]
Qsh = .5*[Qs-Qs' Ef'*Bs;
         -Bs*Ef  Bs]
Vh = [Vq;Vf]
Ph = M\transpose(Vh)
VhP = Vh*Pq

# make sparse skew symmetric versions of the operators"
# precompute union of sparse ids for Qr, Qs
Qrhskew = .5*(Qrh-transpose(Qrh))
Qshskew = .5*(Qsh-transpose(Qsh))

# pack SBP operators into tuple
@unpack LIFT = rd
ops = (Qrhskew,Qshskew,VhP,Vh,Ph,LIFT,Vq)

# interpolate geofacs to both vol/surf nodes
@unpack rxJ, sxJ, ryJ, syJ = md
rxJ, sxJ, ryJ, syJ = (x->Vh*x).((rxJ, sxJ, ryJ, syJ)) # interp to hybridized points
@pack! md = rxJ, sxJ, ryJ, syJ

## subcell versions

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
        #
        h2 = (wf[i]/sum(wf))*2/pi # set so that h = radius of circle with area w_i = face weight
        nnbrs = 3 #min(4,max(3,count(d2[p] .< h2))) # find 3 closest points
        p = p[1:nnbrs]
        Ei = vandermonde_2D(1,[rf[i]],[sf[i]])/vandermonde_2D(1,rq[p],sq[p])
        E[i,p] = Ei
    end
    E = Matrix(droptol!(sparse(E),1e-14))

    # build stencil
    A = spzeros(length(req),length(req))
    for i = 1:length(req)
        d2 = @. (req-req[i])^2 + (seq-seq[i])^2
        p = sortperm(d2)

        # h^2 = wq[i]/pi = radius of circle with area wq[i]
        # h2 =     (sqrt(3)/sum(wq))*wq[i]/pi
        h2 = α^2*(sqrt(3)/sum(wq))*wq[i]/pi

        nnbrs = count(d2[p] .< h2)
        nbrs = p[1:nnbrs]
        A[i,nbrs] .= one(eltype(A))
    end
    A = (A+A')
    A.nzval .= one(eltype(A)) # bool-ish

    # build graph Laplacian
    L1 = (A-diagm(diag(A))) # ignore
    L1 -= diagm(vec(sum(L1,dims=2)))

    b1r = -sum(.5*E'*Br*E,dims=2)
    b1s = -sum(.5*E'*Bs*E,dims=2)
    ψ1r = pinv(L1)*b1r
    ψ1s = pinv(L1)*b1s

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

@unpack rq,sq,wq,rf,sf,wf,nrJ,nsJ = rd
α = 2.33 #5
Qr0,Qs0,E0,_ = build_meshfree_sbp(rq,sq,wq,rf,sf,wf,nrJ,nsJ,α)

Qrh0 = .5*[Qr0-Qr0' E0'*Br;
         -Br*E0  Br]
Qsh0 = .5*[Qs0-Qs0' E0'*Bs;
         -Bs*E0  Bs]
Vh0 = [eye(length(rq));E0]*Vq
LIFT0 = M\(Vq'*E0'*diagm(wf))

# Vh0 = [Vq;Vf]
# LIFT0 = copy(LIFT)

Ph0 = M\transpose(Vh0)
VhP0 = Vh0*Pq


# make sparse skew symmetric versions of the operators"
# precompute union of sparse ids for Qr, Qs
Qrhskew0 = .5*(Qrh0-transpose(Qrh0))
Qshskew0 = .5*(Qsh0-transpose(Qsh0))

opsLow = (Qrhskew0,Qshskew0,VhP0,Vh0,Ph0,LIFT0,Vq)

## rhs functions

"dense version - speed up by prealloc + transpose for col major "
function dense_hadamard_sum(Qhe,ops,opsLow,vgeo,flux_fun::F1,dissipation_fun::F2,skip) where {F1,F2}

    (Qr,Qs) = ops
    (rxJ,sxJ,ryJ,syJ) = vgeo

    # transpose for column-major evals
    QxTr = transpose(rxJ*Qr + sxJ*Qs)
    QyTr = transpose(ryJ*Qr + syJ*Qs)

    QrLow,QsLow = opsLow
    QxTrLow = transpose(rxJ*QrLow + sxJ*QsLow)
    QyTrLow = transpose(ryJ*QrLow + syJ*QsLow)

    n = size(Qr,1)
    nfields = length(Qhe)

    QF = zero.(Qhe) #ntuple(x->zeros(n),nfields)
    QFi = zeros(nfields)
    F = (zeros(nfields),zeros(nfields)) # two flux outputs in 2D
    for i = 1:n
        Qi = getindex.(Qhe,i)
        fill!(QFi,0)
        for j = 1:n
            if (i > skip & j > skip)==false
                Qj = getindex.(Qhe,j)
                Fx,Fy = flux_fun(Qi,Qj)
                Dx,Dy = dissipation_fun(Qi,Qj)
                diss = @. .0*(abs(QxTrLow[i,j])*Dx + abs(QyTrLow[i,j])*Dy)
                @. QFi += QxTr[j,i]*Fx + QyTr[j,i]*Fy - diss
            end
        end

        for fld in eachindex(Qhe)
            QF[fld][i] = QFi[fld]
        end
    end

    return QF
end


"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs(Q,md::MeshData,ops,opsLow,flux_fun::F1,dissipation_fun::F2) where {F1,F2}

    @unpack rxJ,sxJ,ryJ,syJ,J,wJq = md
    @unpack nxJ,nyJ,sJ,mapP,mapB,K = md
    Qrh,Qsh,VhP,Vh,Ph,Lf,Vq = ops
    QrhLow,QshLow,_ = opsLow
    Nh,Nq = size(VhP)

    # interpolate + compute face values
    Qh = (x->Vh*x).(Q)
    QM = (x->x[Nq+1:Nh,:]).(Qh)
    QP = (x->x[mapP]).(QM)

    # simple lax friedrichs dissipation
    LFc = .25 .* sJ
    fSx,fSy = flux_fun(QM,QP)
    normal_flux(fx,fy,u) = fx.*nxJ + fy.*nyJ - LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,fSy,QM)
    rhsQ = (x->Lf*x).(flux)

    mxm_accum!(X,x,e) = X[:,e] .+= 2*Ph*x

    # compute volume contributions using flux differencing
    for e = 1:K
        Qhe = getindex.(Qh,:,e) # force tuples for fast splatting
        vgeo_local = getindex.((rxJ,sxJ,ryJ,syJ),1,e) # assumes affine elements for now

        Qops = (Qrh,Qsh)
        QopsLow = (QrhLow,QshLow)
        QFe = dense_hadamard_sum(Qhe,Qops,QopsLow,vgeo_local,flux_fun,dissipation_fun,Nq)

        mxm_accum!.(rhsQ,QFe,e)
    end

    rhstest = sum(Q[1].*(rd.M*rhsQ[1]))
    rhsQ = (x -> -x./J).(rhsQ)
    return rhsQ,rhstest # scale by Jacobian
end

function advec_flux(QL,QR)
    # uL = QL[1]
    # uR = QR[1]
    return ((x,y)->.5*(x+y)).(QL,QR), zero.(QL) #@SVector [(@. .5*(uL+uR))]
end

function advec_diss(QL,QR)
    return ((x,y)->.5*(y-x)).(QL,QR), zero.(QL) #@SVector [(@. .5*(uL+uR))]
end

## Time integration
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant for CFL
h  = 2/K1D
dt = CFL * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# define initial conditions at nodes
@unpack x,y = md
uinit(x,y) = @. sin(pi*x)*sin(pi*y)
uinit(x,y) = @. exp(-25*(x^2+y^2))
u = uinit(x,y)
# u = Pq*((@. abs(md.xq) < .5) .& (@. abs(md.yq) < .5))
Q = tuple(u)
resQ = zero.(Q)

for i = 1:Nsteps

    rhstest = 0
    for INTRK = 1:5
        rhsQ,rhstest = rhs(Q,md,ops,opsLow,advec_flux,advec_diss)
        bcopy!.(resQ, @. rk4a[INTRK]*resQ + dt*rhsQ)
        bcopy!.(Q, @. Q + rk4b[INTRK]*resQ)
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps: rhstest = $rhstest")
    end
end

# use a higher degree quadrature for error evaluation
@unpack VDM = rd
@unpack J = md
rq2,sq2,wq2 = quad_nodes_2D(N+2)
Vq2 = vandermonde_2D(N,rq2,sq2)/VDM
wJq2 = diagm(wq2)*(Vq2*J)
xq2,yq2 = (x->Vq2*x).((x,y))

Qq = (x->Vq2*x).(Q)
Qex = [uinit(xq2 .- T,yq2)]
L2err = 0.0
for fld in eachindex(Q)
    global L2err
    L2err += sum(@. wJq2*(Qq[fld]-Qex[fld])^2)
end
L2err = sqrt(L2err)
println("L2err at final time T = $T is $L2err\n")

#plotting nodes
rp, sp = equi_nodes_2D(35)
Vp = vandermonde_2D(N,rp,sp)/VDM

gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

vv,xp,yp = (x->Vp*x).((Q[1],x,y))
scatter(xp,yp,vv,zcolor=vv,camera=(0,90))
