using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

"Approximation parameters"
N = 2 # The order of approximation
K1D = 12
CFL = 2 # CFL goes up to 2.5ish
T = 1.0 # endtime

"Mesh related variables"
Kx = convert(Int,4/3*K1D)
Ky = K1D
VX, VY, EToV = uniform_tri_mesh(Kx, Ky)
@. VX = 15*(1+VX)/2
@. VY = 5*VY

# initialize ref element and mesh
# specify lobatto nodes to automatically get DG-SEM mass lumping
rd = init_reference_tri(N)
md = init_mesh((VX,VY),EToV,rd)

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

# construct hybridized SBP operators
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

# define initial conditions at nodes
@unpack x,y = md
rho,u,v,p = vortex(x,y,0)
Q = primitive_to_conservative(rho,u,v,p)

# interpolate geofacs to both vol/surf nodes
@unpack rxJ, sxJ, ryJ, syJ = md
rxJ, sxJ, ryJ, syJ = (x->Vh*x).((rxJ, sxJ, ryJ, syJ)) # interp to hybridized points
@pack! md = rxJ, sxJ, ryJ, syJ

# pack SBP operators into tuple
@unpack LIFT = rd
ops = (Qrhskew,Qshskew,VhP,Ph,LIFT,Vq)

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant for CFL
h  = 2/K1D
dt = CFL * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"dense version - speed up by prealloc + transpose for col major "
function dense_hadamard_sum(Qhe,ops,vgeo,flux_fun)

    (Qr,Qs) = ops
    (rxJ,sxJ,ryJ,syJ) = vgeo
    # transpose for column-major evals
    QxTr = transpose(rxJ*Qr + sxJ*Qs)
    QyTr = transpose(ryJ*Qr + syJ*Qs)

    # precompute logs for logmean
    (rho,u,v,beta) = Qhe
    Qlog = (log.(rho), log.(beta))
    Qlogi = zeros(length(Qlog))
    Qlogj = zeros(length(Qlog))

    n = size(Qr,1)
    nfields = length(Qhe)

    QF = ntuple(x->zeros(n),nfields)
    QFi = zeros(nfields)
    for i = 1:n
        Qi = (x->x[i]).(Qhe)
        Qlogi = (x->x[i]).(Qlog)

        fill!(QFi,0)
        for j = 1:n
            Qj = (x->x[j]).(Qhe)
            Qlogj = (x->x[j]).(Qlog)

            Fx,Fy = flux_fun(Qi,Qj,Qlogi,Qlogj)
            @. QFi += QxTr[j,i]*Fx + QyTr[j,i]*Fy
        end

        for fld in eachindex(Qhe)
            QF[fld][i] = QFi[fld]
        end
    end

    return QF
end


"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs(Q,md::MeshData,ops,flux_fun,compute_rhstest=false)

    @unpack rxJ,sxJ,ryJ,syJ,J,wJq = md
    @unpack nxJ,nyJ,sJ,mapP,mapB,K = md
    Qrh,Qsh,VhP,Ph,Lf,Vq = ops
    Nh,Nq = size(VhP)

    # entropy var projection
    VU = v_ufun((x->Vq*x).(Q)...)
    VU = (x->VhP*x).(VU)
    Uh = u_vfun(VU...)

    # convert to rho,u,v,beta vars
    (rho,rhou,rhov,E) = Uh
    beta = betafun(rho,rhou,rhov,E)
    Qh = (rho, rhou./rho, rhov./rho, beta) # redefine Q = (rho,u,v,β)

    # compute face values
    QM = (x->x[Nq+1:end,:]).(Qh)
    QP = (x->x[mapP]).(QM)

    # simple lax friedrichs dissipation
    Uf =  (x->x[Nq+1:end,:]).(Uh)
    (rhoM,rhouM,rhovM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    lam = abs.(wavespeed(rhoM,rhoUM_n,EM))
    LFc = .5*max.(lam,lam[mapP]).*sJ

    fSx,fSy = flux_fun(QM,QP)
    normal_flux(fx,fy,u) = fx.*nxJ + fy.*nyJ - LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,fSy,Uf)
    rhsQ = (x->Lf*x).(flux)

    # compute volume contributions using flux differencing
    for e = 1:K
        Qhe = tuple((x->x[:,e]).(Qh)...) # force tuples for fast splatting
        vgeo_local = (x->x[1,e]).((rxJ,sxJ,ryJ,syJ)) # assumes affine elements for now

        Qops = (Qrh,Qsh)
        QFe = dense_hadamard_sum(Qhe,Qops,vgeo_local,flux_fun)

        mxm_accum!(X,x) = X[:,e] += 2*Ph*x
        mxm_accum!.(rhsQ,QFe)
    end

    rhsQ = (x -> -x./J).(rhsQ)

    rhstest = 0
    if compute_rhstest
        for fld in eachindex(rhsQ)
            VUq = VU[fld][1:Nq,:]
            rhstest += sum(wJq.*VUq.*(Vq*rhsQ[fld]))
        end
    end

    return rhsQ,rhstest # scale by Jacobian
end

Q = collect(Q) # make Q,resQ arrays of arrays for mutability
resQ = [zeros(size(x)) for i in eachindex(Q)]

# # testing
# rhsQ,rhstest = rhs(Q,md,ops,euler_fluxes,true)
# println("Testing: rhstest = $rhstest")


for i = 1:Nsteps

    rhstest = 0
    for INTRK = 1:5
        compute_rhstest = true # INTRK==5
        rhsQ,rhstest = rhs(Q,md,ops,euler_fluxes,compute_rhstest)

        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")
    end
end

# "project solution back to GLL nodes"
# Q = (x->Pq*x).(Q)

# use a higher degree quadrature for error evaluation
@unpack VDM = rd
@unpack J = md
rq2,sq2,wq2 = quad_nodes_2D(N+2)
Vq2 = vandermonde_2D(N,rq2,sq2)/VDM
wJq2 = diagm(wq2)*(Vq2*J)
xq2,yq2 = (x->Vq2*x).((x,y))

Qq = (x->Vq2*x).(Q)
Qex = primitive_to_conservative(vortex(xq2,yq2,T)...)
L2err = 0.0
for fld in eachindex(Q)
    global L2err
    L2err += sum(@. wJq2*(Qq[fld]-Qex[fld])^2)
end
L2err = sqrt(L2err)
println("L2err at final time T = $T is $L2err\n")

#plotting nodes
@unpack Vp = rd
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

vv = Vp*Q[1]
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
