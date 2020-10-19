using BenchmarkTools
using LoopVectorization
using LinearAlgebra
using Plots
using UnPack

using EntropyStableEuler.Fluxes2D
import EntropyStableEuler.Fluxes1D: wavespeed_1D # for LF flux
import EntropyStableEuler: γ
using FluxDiffUtils
using StartUpDG
using StartUpDG.ExplicitTimestepUtils

include("HybridizedSBPUtils.jl")
using .HybridizedSBPUtils

N = 3
K1D = 8
T = .25
CFL = .5

# init ref element and mesh
rd = init_reference_tri(N)
VX,VY,EToV = uniform_tri_mesh(K1D)
md = init_DG_mesh(VX,VY,EToV,rd)

# Make domain periodic
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
build_periodic_boundary_maps!(md,rd,LX,LY)

#####
##### Define initial coefficients and time-stepping
#####

# "Define the initial conditions by interpolation"
@unpack x,y = md
rho = @. 2 + .5*exp(-100*(x^2+y^2))
u = @. 0*x
v = @. 0*x
p = @. rho^γ

Q = primitive_to_conservative(rho,u,v,p)
Qrhskew,Qshskew,Vh,Ph,VhP = build_hSBP_ops(rd)
SBP_ops = (Matrix(Qrhskew'),Matrix(Qshskew'),Vh,Ph,VhP)

# interpolate geofacs to both vol/surf nodes, pack back into md
@unpack Vq,Vf = rd
@unpack rxJ, sxJ, ryJ, syJ = md
rxJ, sxJ, ryJ, syJ = (x->[Vq;Vf]*x).((rxJ, sxJ, ryJ, syJ)) # interp to hybridized points
@pack! md = rxJ, sxJ, ryJ, syJ

# e = 1
# Qh = (x->Vh*x).(Q)
# Qh_log = (Qh...,log.(first(Qh)),log.(last(Qh)))
# rhse = zero.(getindex.(Qh,:,e))
# hadamard_sum!(rhse,(Qrhskew,Qshskew),euler_fluxes,getindex.(Qh_log,:,e))
# rhse_old = dense_hadamard_sum(getindex.(Qh,:,e),(Qrhskew,Qshskew),euler_fluxes)
# @show sum(norm.(rhse .- rhse_old))
#
# function time_had()
#     hadamard_sum!(rhse,(Qrhskew,Qshskew),euler_fluxes,getindex.(Qh_log,:,e))
# end

# # old version
# function dense_hadamard_sum(Qhe,ops,flux_fun)
#
#     (QxTr,QyTr) = ops
#
#     # precompute logs for logmean
#     (rho,u,v,beta) = Qhe
#     Qlog = (log.(rho), log.(beta))
#
#     n = size(QxTr,1)
#     nfields = length(Qhe)
#
#     QF = zero.(Qhe) #ntuple(x->zeros(n),nfields)
#     QFi = zeros(nfields)
#     for i = 1:n
#         Qi = getindex.(Qhe,i)
#         Qlogi = getindex.(Qlog,i)
#
#         fill!(QFi,0)
#         for j = 1:n
#             Qj = getindex.(Qhe,j)
#             Qlogj = getindex.(Qlog,j)
#
#             Fx,Fy = flux_fun(Qi...,Qlogi...,Qj...,Qlogj...)
#             @. QFi += QxTr[j,i]*Fx + QyTr[j,i]*Fy
#         end
#
#         for field in eachindex(Qhe)
#             QF[field][i] = QFi[field]
#         end
#     end
#
#     return QF
# end

# function time_had2()
#     dense_hadamard_sum(getindex.(Qh,:,e),(Qrhskew,Qshskew),euler_fluxes)
# end
# @btime time_had()
# @btime time_had2()

# convert to rho,u,v,beta vars
function cons_to_prim_withlogs(rho,rhou,rhov,E)
    rho,u,v,beta = conservative_to_primitive_beta(rho,rhou,rhov,E)
    return (rho,u,v,beta),(log.(rho),log.(beta))
end

mxm_accum!(X,x,e) = X[:,e] .+= 2*Ph*x

function rhs(Q, SBP_ops, rd::RefElemData, md::MeshData)

    QrTr,QsTr,Vh,Ph,VhP = SBP_ops
    @unpack LIFT,Vf,Vq = rd
    @unpack rxJ,sxJ,ryJ,syJ,J,K = md
    @unpack nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    Nh,Nq = size(VhP)

    # entropy var projection
    VU = (x->VhP*x).(v_ufun((x->Vq*x).(Q)...))
    Uh = u_vfun(VU...)
    Qh,logQh = cons_to_prim_withlogs(Uh...)
    Qh_log = (Qh...,logQh...)

    # compute face values
    QM = (x->x[Nq+1:Nh,:]).(Qh_log)
    QP = (x->x[mapP]).(QM)

    # simple lax friedrichs dissipation
    Uf =  (x->x[Nq+1:Nh,:]).(Uh)
    (rhoM,rhouM,rhovM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    lam = abs.(wavespeed_1D(rhoM,rhoUM_n,EM))
    LFc = .25*max.(lam,lam[mapP]).*sJ

    fSx,fSy = euler_fluxes(QM...,QP...)
    normal_flux(fx,fy,u) = fx.*nxJ + fy.*nyJ - LFc.*(u[mapP]-u)
    flux = map(normal_flux,fSx,fSy,Uf)
    rhsQ = (x->LIFT*x).(flux)

    # compute volume contributions using flux differencing
    rhse = zero.(getindex.(Qh,:,1))
    for e = 1:K
        QxTr = rxJ[1,e]*QrTr + sxJ[1,e]*QsTr
        QyTr = ryJ[1,e]*QrTr + syJ[1,e]*QsTr
        # hadamard_sum!(rhse,(QxTr,QyTr),euler_fluxes,getindex.(Qh_log,:,e))

        # will compute sum(Ax.*Fx + Ay.*Fy,dims=2)
        # euler_fluxes(QL_log_e...,QR_log_e...)
        hadamard_sum!(rhse,(QxTr,QyTr),euler_fluxes,getindex.(Qh_log,:,e);
                      skip_index=(i,j)->(i>Nq)&(j>Nq))
        # dense_hadamard_sum(getindex.(Qh,:,e),(QxTr,QyTr),euler_fluxes)
        mxm_accum!.(rhsQ,rhse,e)
    end

    rhstest = zero(eltype(first(rhsQ)))
    compute_rhstest = false
    if compute_rhstest
        for fld in eachindex(rhsQ)
            VUq = VU[fld][1:Nq,:]
            rhstest += sum(md.wJq.*VUq.*(Vq*rhsQ[fld]))
        end
    end

    return map(x -> -x./J,rhsQ),rhstest
end

# rhsQ,rhstest = rhs(Q,SBP_ops,rd,md)
# error("d")

# "Time integration coefficients"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# "Perform time-stepping"
resQ = zero.(Q)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ,rhstest = rhs(Q,SBP_ops,rd,md)
        bcopy!.(resQ, @. rk4a[INTRK]*resQ + dt*rhsQ)
        bcopy!.(Q,@. Q + rk4b[INTRK]*resQ)
    end

    if i%10==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
    end
end

# plotting nodes
gr(aspect_ratio=1,legend=false)
scatter((x->rd.Vp*x).((x,y,Q[1])),zcolor=rd.Vp*Q[1],msw=0,ms=2,cam=(0,90))

#
#
# # dopri initialization
# rka,rkE,rkc = dp56()
# PIparams = init_PI_controller(5)
# Qtmp = similar.(Q)
# rhsQrk = ntuple(x->zero.(Q),length(rkE))
# prevErrEst = nothing
# rhsQ = rhs_avx(Q,rd,md)
# bcopy!.(rhsQrk[1],rhsQ) # initialize DOPRI rhs (FSAL property)
#
# t = 0.0
# i = 0
# dthist = [dt]
#
# while t < T
#     for INTRK = 2:7
#         k = zero.(Qtmp)
#         for s = 1:INTRK-1
#             bcopy!.(k, @. k + rka[INTRK,s]*rhsQrk[s])
#         end
#         bcopy!.(Qtmp, @. Q + dt*k)
#         rhsQ = rhs_avx(Qtmp,rd,md)
#         bcopy!.(rhsQrk[INTRK],rhsQ)
#     end
#
#     global t,dt,i,prevErrEst
#     accept_step, dtnew, prevErrEst
#         = compute_adaptive_dt(Q,rhsQrk,dt,rkE,PIparams,prevErrEst)
#     if accept_step
#         t += dt
#         bcopy!.(Q, Qtmp)
#         bcopy!.(rhsQrk[1], rhsQrk[7]) # use FSAL property
#         push!(dthist,dt) # store dt history
#     end
#     dt = min(T-t,dtnew)
#     i = i + 1  # number of total steps attempted
#
#     if i%10==0
#         println("i = $i, t = $t, dt = $dtnew, errEst = $prevErrEst")
#     end
# end
#
# # plotting nodes
# gr(aspect_ratio=1,legend=false)
# scatter((x->rd.Vp*x).((x,y,Q[1])),zcolor=rd.Vp*Q[1], msw=0,ms=2,cam=(0,90))
