using Plots

using UnPack
using Setfield
using StaticArrays

using RecursiveArrayTools

using LinearAlgebra
using NodesAndModes
using StartUpDG
using EntropyStableEuler
using FluxDiffUtils

N = 2
K1D = 200
CFL = .5
T = .2
interval = 50

#######################################################
#           problem setup                             #
#######################################################

struct Sod end
struct MHD end
problem = Sod()
# problem = MHD()

# speed of sound in general dims
function cfun(U)
    eqn = Euler{1}()
    p = pfun(eqn,U)
    return @. sqrt(p*eqn.γ/U[1])
end

# sod
function u0(x,problem::Sod)
    ρ(x) = (x < 0) ? 1.0 : .125
    u(x) = 0.0
    p(x) = (x < 0) ? 1.0 : .1
    return ρ.(x),u.(x),p.(x)
end
add_source!(rhsQ,Q,md,problem::Sod) = return nothing
function impose_BCs!(UP,problem::Sod)
    UBC_left = prim_to_cons(Euler{1}(),u0(-1,problem))
    UBC_right = prim_to_cons(Euler{1}(),u0(1,problem))
    # outflow pressure
    for fld = 1:length(UP)
        UP[fld][1] = UBC_left[fld]
        UP[fld][end] = UBC_right[fld]
    end
end
make_domain(VX,problem::Sod) = VX

# MHD source terms
function u0(x,problem::MHD)
    ρ(x) = 3.92427020*2.883061e-04
    u(x) = 0.45157424/13.88*sqrt(61.3483371)*4000
    # p = ρRT.
    T(x) = 61.3483371*2.391387e+02
    R = 8.3145 # gas constant
    p(x) = ρ(x)*R*T(x) # set based on Mach 13.7
    return ρ.(x),u.(x),p.(x)
end
function add_source!(rhsQ,Q,md,problem::MHD)
    @unpack x = md
    u = @. Q[2]/Q[1]
    σ = 1.0
    Ez = 10000.0
    By = 1.0
    @. rhsQ[2] += σ*(Ez-By*u)*By
    @. rhsQ[3] += σ*(Ez-By*u)*Ez
    return nothing
end
function impose_BCs!(UP,problem::MHD)

    ρ,ρu,E = UP

    UBC_left = prim_to_cons(Euler{1}(),u0(-1,problem))
    UBC_right = prim_to_cons(Euler{1}(),u0(1,problem))
    for fld = 1:length(UP)
        UP[fld][1] = UBC_left[fld]
        UP[fld][end] = UBC_right[fld]
    end
end
make_domain(VX,problem::MHD) = VX = @. .5*(1+VX)*.125

###########################################
#      build mesh,ref elem,etc
###########################################

rd = RefElemData(Line(),N)
VX,EToV = uniform_mesh(Line(),K1D)
VX = make_domain(VX,problem)
md = MeshData(VX,EToV,rd)

# md = make_periodic(md,rd)

###########################################
#      build hybridized SBP operators
###########################################

@unpack M,Dr,Vq,Pq,Vf,wf,nrJ = rd
Qr = Pq'*M*Dr*Pq
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
        -Br*Ef  Br]
Vh = [Vq;Vf]
Ph = M\transpose(Vh)
VhP = Vh*Pq

# make skew symmetric versions of the operators"
Qrhskew = .5*(Qrh-transpose(Qrh))
SBP_ops = (Matrix(Qrhskew'),Ph,VhP)

# @unpack rxJ = md # hack - rxJ = 1 here, ignore

#######################################################
#                   initialize u0 and Q               #
#######################################################

@unpack x,xq = md
rho,u,p = (x->Pq*x).(u0(xq,problem))

Q = VectorOfArray(prim_to_cons(Euler{1}(),SVector{3}(rho,u,p)))

#######################################################
#               local rhs computations                #
#######################################################

function rhs(Q, SBP_ops, rd::RefElemData, md::MeshData, problem)

    QrTr,Ph,VhP = SBP_ops
    Ph *= 2.0
    @unpack LIFT,Vf,Vq = rd
    @unpack rxJ,J,K = md
    @unpack nxJ,sJ = md
    @unpack mapP,mapB = md

    Nh,Nq = size(VhP)

    # entropy var projection
    VU = (x->VhP*x).(v_ufun(Euler{1}(),(x->Vq*x).(Q.u)))
    Uh = u_vfun(Euler{1}(),VU)

    # compute face values
    Uf = (x->x[Nq+1:Nh,:]).(Uh)
    UP = (x->x[mapP]).(Uf)
    impose_BCs!(UP,problem)

    # simple lax friedrichs dissipation
    (rhoM,rhouM,EM) = Uf
    lam = abs.(rhouM./rhoM) + sqrt.(Euler{1}().γ*pfun(Euler{1}(),Uf)./rhoM)
    LFc = .25*max.(lam,lam[mapP]).*sJ

    fSx = fS(Euler{1}(),Uf,UP)
    normal_flux(fx,u) = @. fx*nxJ - LFc*(u[mapP]-u)
    flux = map(normal_flux,fSx,Uf)
    rhsQ = (x->LIFT*x).(flux)

    # compute volume contributions using flux differencing
    rhse = zero.(getindex.(Uh,:,1))
    for e = 1:K
        fill!.(rhse,0.0)
        hadamard_sum_ATr!(rhse, (QrTr,), (x,y)->tuple(fS(Euler{1}(),x,y)),
                          getindex.(Uh,:,e); skip_index=(i,j)->(i>Nq)&&(j>Nq))
        for fld = 1:length(rhse)
            rhsQ[fld][:,e] .+= Ph*rhse[fld]
        end
    end

    return VectorOfArray(map(x -> -x./J,rhsQ))
end

# "Time integration coefficients"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)^2/2  # estimated trace constant

λ = maximum(abs.(Q[2]./Q[1]) .+ cfun(Q))
dt = CFL * 2 / (λ*CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# "Perform time-stepping"
resQ = zero(Q)
#@gif
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,SBP_ops,rd,md,problem)
        add_source!(rhsQ,Q,md,problem)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q   += rk4b[INTRK]*resQ
    end

    if i%interval==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
        # scatter((x->rd.Vp*x).((x,Q[1])),zcolor=rd.Vp*Q[1],msw=0,ms=2,cam=(0,90))
    end
end #every interval

scatter((x->rd.Vp*x).((x,Q[1])),msw=0,ms=1.5,leg=false)
