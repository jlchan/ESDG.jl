using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

"Approximation parameters"
N = 4 # The order of approximation
K1D = 12
CFL = 2 # CFL goes up to 3 OK...
T = 5.0 # endtime

"Mesh related variables"
Kx = convert(Int,4/3*K1D)
Ky = K1D
(VX, VY, EToV) = uniform_quad_mesh(Kx, Ky)
@. VX = 15*(1+VX)/2
@. VY = 5*VY

# initialize ref element and mesh
# specify lobatto nodes to automatically get DG-SEM mass lumping
rd = init_reference_quad(N,gauss_quad(0,0,N))
md = init_mesh_2D((VX,VY),EToV,rd)

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

#Make hybridized SBP operators
@unpack M,Dr,Ds,Vq,Pq,Vf,wf,nrJ,nsJ = rd
Qr = Pq'*M*Dr*Pq
Qs = Pq'*M*Ds*Pq
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Bs = diagm(wf.*nsJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
-Br*Ef Br]
Qsh = .5*[Qs-Qs' Ef'*Bs;
-Bs*Ef Bs]

# make sparse skew symmetric versions of the operators"
Qrhskew = .5*(Qrh-transpose(Qrh))
Qshskew = .5*(Qsh-transpose(Qsh))
Qrh_sparse = droptol!(sparse(Qrhskew),1e-12)
Qsh_sparse = droptol!(sparse(Qshskew),1e-12)

# precompute union of sparse ids for Qr, Qs
Qrsids = [unique([Qrh_sparse[i,:].nzind; Qsh_sparse[i,:].nzind]) for i = 1:size(Qrhskew,1)]

# Make node maps periodic
@unpack Nfaces = rd
@unpack x,y,xf,yf,K,mapM,mapP,mapP = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY))
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

# define initial conditions by interpolation
rho,u,v,p = vortex(x,y,0)
Q = primitive_to_conservative(rho,u,v,p)

# convert operators to a quadrature-node basis
@unpack wq = rd
Vh = droptol!(sparse([eye(length(wq)); Ef]),1e-12)
Ph = droptol!(sparse(2*diagm(@. 1/wq)*transpose(Vh)),1e-12)
Lf = droptol!(sparse(diagm(@. 1/wq)*(transpose(Ef)*diagm(wf))),1e-12)
Q = (x->Vq*x).(Q)

# Pack arguments into tuples
ops = (Qrhskew,Qshskew,Qrh_sparse,Qsh_sparse,Qrsids,Ph,Lf)
@unpack rxJ, sxJ, ryJ, syJ, J, nxJ, nyJ, sJ = md
vgeo = (x->Vh*x).((rxJ, sxJ, ryJ, syJ, J)) # interp to hybridized points
fgeo = (nxJ,nyJ,sJ)
nodemaps = (mapP,mapB)

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

"sparse version - precompute sparse row ids for speed"
function sparse_hadamard_sum(Qhe,ops,vgeo,flux_fun)

    (Qr,Qs,Qnzids) = ops
    (rxJ,sxJ,ryJ,syJ) = vgeo
    nrows = size(Qr,1)
    nfields = length(Qhe)

    # precompute logs for logmean
    (rho,u,v,beta) = Qhe
    Qlog = (log.(rho), log.(beta))

    rhsQe = ntuple(x->zeros(nrows),nfields)
    rhsi = zeros(nfields) # prealloc a small array
    for i = 1:nrows
        Qi = (x->x[i]).(Qhe)
        Qlogi = (x->x[i]).(Qlog)

        fill!(rhsi,0) # reset rhsi before accumulation
        for j = Qnzids[i] # nonzero row entries
            Qj = (x->x[j]).(Qhe)
            Qlogj = (x->x[j]).(Qlog)

            Fx,Fy = flux_fun(Qi,Qj,Qlogi,Qlogj)
            Fr = @. rxJ*Fx + ryJ*Fy
            Fs = @. sxJ*Fx + syJ*Fy

            @. rhsi += Qr[i,j]*Fr + Qs[i,j]*Fs
        end

        # faster than one-line fixes (no return args)
        for fld in eachindex(rhsQe)
            rhsQe[fld][i] = rhsi[fld]
        end
    end

    return rhsQe
end

"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs(Qh,UM,ops,vgeo,fgeo,nodemaps,flux_fun)

    # unpack args
    Qrh,Qsh,Qrh_sparse,Qsh_sparse,Qrsids,Ph,Lf=ops
    nxJ,nyJ,sJ = fgeo
    mapP,mapB = nodemaps
    Nh = size(Qrhskew,1)
    Nq = size(Ph,1)
    K  = size(Qh[1],2)

    QM = (x->x[Nq+1:end,:]).(Qh)
    QP = (x->x[mapP]).(QM)

    # simple lax friedrichs dissipation
    (rhoM,rhouM,rhovM,EM) = UM
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    lam = abs.(wavespeed(rhoM,rhoUM_n,EM))
    LFc = .5*max.(lam,lam[mapP]).*sJ

    fSx,fSy = flux_fun(QM,QP)
    normal_flux(fx,fy,u) = fx.*nxJ + fy.*nyJ - LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,fSy,UM)
    rhsQ = (x->Lf*x).(flux)

    # compute volume contributions using flux differencing
    for e = 1:K
        Qhe = tuple((x->x[:,e]).(Qh)...) # force tuples for splatting speed
        vgeo_local = (x->x[1,e]).(vgeo) # assumes affine elements for now

        Qops = (Qrh_sparse,Qsh_sparse,Qrsids)
        QFe = sparse_hadamard_sum(Qhe,Qops,vgeo_local,flux_fun)

        # Qops = (Qrh,Qsh)
        # QFe = dense_hadamard_sum(Qhe,Qops,vgeo,flux_fun)

        applyA!(X,x,e) = X[:,e] += Ph*x
        applyA!.(rhsQ,QFe,e)
    end

    return (x -> -x./J).(rhsQ) # scale by Jacobian
end

@unpack wJq = md
Q = collect(Q) # make Q,resQ arrays of arrays for mutability
resQ = [zeros(size(x)) for i in eachindex(Q)]
for i = 1:Nsteps

    rhstest = 0
    for INTRK = 1:5
        compute_rhstest = INTRK==5

        VU = v_ufun(Q...)
        Uf = u_vfun((x->Ef*x).(VU)...)
        (rho,rhou,rhov,E) = vcat.(Q,Uf) # assume volume nodes are collocated

        # convert to rho,u,v,beta vars
        beta = betafun(rho,rhou,rhov,E)
        Qh = (rho,rhou./rho,rhov./rho,beta) # redefine Q = (rho,u,v,β)

        rhsQ = rhs(Qh,Uf,ops,vgeo,fgeo,nodemaps,euler_fluxes)

        if compute_rhstest
            for fld in eachindex(rhsQ)
                rhstest += sum(wJq.*VU[fld].*rhsQ[fld])
            end
        end

        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")
    end
end

"project solution back to GLL nodes"
Q = (x->Pq*x).(Q)

"higher degree quadrature for error evaluation"

@unpack VDM = rd
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

# pyplot(size=(200,200),legend=false,markerstrokewidth=0,markersize=2)
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

vv = Vp*Q[1]
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
