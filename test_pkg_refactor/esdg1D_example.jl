using Plots
using UnPack
using LinearAlgebra

using StartUpDG
using StartUpDG.ExplicitTimestepUtils

using EntropyStableEuler.Fluxes1D
import EntropyStableEuler: γ

using FluxDiffUtils

include("HybridizedSBPUtils.jl")
using .HybridizedSBPUtils

N = 4
K1D = 64
CFL = .75
T = 1.

rd = init_reference_interval(N)
VX,EToV = uniform_1D_mesh(K1D)
md = init_DG_mesh(VX,EToV,rd)

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
SBP_ops = (Matrix(Qrhskew'),Vh,Ph,VhP)

LX = 2
build_periodic_boundary_maps!(md,rd,LX)


#######################################################
#                                                     #
#######################################################

@unpack x = md
rho = @. 1 + .5*exp(-25*(x^2))
u = @. 0*x
p = @. rho^γ

Q = primitive_to_conservative(rho,u,p)

# interpolate geofacs to both vol/surf nodes, pack back into md
@unpack Vq,Vf = rd
@unpack rxJ = md
rxJ = [Vq;Vf]*rxJ # interp to hybridized points
@pack! md = rxJ

#######################################################
#               local rhs computations                #
#######################################################

# convert to rho,u,v,beta vars
function cons_to_prim_withlogs(rho,rhou,E)
    rho,u,beta = conservative_to_primitive_beta(rho,rhou,E)
    Qh=(rho,u,beta)
    Qhlog = (Qh...,log.(rho),log.(beta))
    return Qh,Qhlog
end

mxm_accum!(X,x,e) = X[:,e] .+= 2*Ph*x

function rhs(Q, SBP_ops, rd::RefElemData, md::MeshData)

    QrTr,Vh,Ph,VhP = SBP_ops
    @unpack LIFT,Vf,Vq = rd
    @unpack rxJ,J,K = md
    @unpack nxJ,sJ = md
    @unpack mapP,mapB = md

    Nh,Nq = size(VhP)

    # entropy var projection
    VU = (x->VhP*x).(v_ufun((x->Vq*x).(Q)...))
    Uh = u_vfun(VU...)
    Qh,Qh_log = cons_to_prim_withlogs(Uh...)

    # compute face values
    QM = (x->x[Nq+1:Nh,:]).(Qh_log)
    QP = (x->x[mapP]).(QM)

    # simple lax friedrichs dissipation
    Uf =  (x->x[Nq+1:Nh,:]).(Uh)
    (rhoM,rhouM,EM) = Uf
    rhoUM_n = @. rhouM*nxJ/sJ
    lam = abs.(wavespeed_1D(rhoM,rhoUM_n,EM))
    LFc = .25*max.(lam,lam[mapP]).*sJ

    fSx = euler_fluxes(QM...,QP...)
    normal_flux(fx,u) = fx.*nxJ - LFc.*(u[mapP]-u)
    flux = map(normal_flux,fSx,Uf)
    rhsQ = (x->LIFT*x).(flux)

    # compute volume contributions using flux differencing
    rhse = zero.(getindex.(Qh,:,1))
    for e = 1:K
        # computes sum(Ax.*Fx + Ay.*Fy,dims=2)
        # wrap QxTr, euler_fluxes in tuple to make it look "multidimensional"
        hadamard_sum!(rhse,(rxJ[1,e]*QrTr,),tuple ∘ euler_fluxes,getindex.(Qh_log,:,e);
                      skip_index=(i,j)->(i>Nq)&(j>Nq))
        mxm_accum!.(rhsQ,rhse,e)
    end

    return map(x -> -x./J,rhsQ)
end


# "Time integration coefficients"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)^2/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# "Perform time-stepping"
resQ = zero.(Q)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,SBP_ops,rd,md)
        bcopy!.(resQ, @. rk4a[INTRK]*resQ + dt*rhsQ)
        bcopy!.(Q,@. Q + rk4b[INTRK]*resQ)
    end

    if i%10==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
    end
end

# plotting nodes
gr(aspect_ratio=1,legend=false)
scatter((x->rd.Vp*x).((x,Q[1])),zcolor=rd.Vp*Q[1],msw=0,ms=2,cam=(0,90))
# mean(x) = vec(rd.wq'*rd.Vq*x/2)
# plot(mean.((x,Q[1])),msw=0,ms=4,marker=:dot,cam=(0,90),leg=false)
