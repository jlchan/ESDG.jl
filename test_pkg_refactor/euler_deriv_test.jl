using Plots
using UnPack
using LinearAlgebra
using SparseArrays

using ForwardDiff
using StaticArrays

using StartUpDG
using StartUpDG.ExplicitTimestepUtils

using EntropyStableEuler.Fluxes1D
import EntropyStableEuler: γ
using FluxDiffUtils

# include("HybridizedSBPUtils.jl")
# using .HybridizedSBPUtils

# using AdjRRK

N = 2
K1D = 2
CFL = .25
T = .5
Nfields = 3

################################################################################
## Initial DG stuff

rd = init_reference_interval(N)
VX,EToV = uniform_1D_mesh(K1D)
md = init_DG_mesh(VX,EToV,rd)

@unpack M,Dr,Vq,Pq,Vf,wf,nrJ = rd

# # collocation
# M = diagm(vec(sum(M,dims=2)))
# Pq = I(N+1)
# Vq = I(N+1)
# Vf = zeros(2,N+1); Vf[1,1] = 1; Vf[2,N+1] = 1

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

################################################################################
##

@unpack x = md
#initial condition
rho = @. 1 + .25*exp(-25*(x^2))
u = @. 0*x + .1
p = @. rho^γ

Q0 = primitive_to_conservative(rho,u,p)
Q = primitive_to_conservative(rho,u,p)

# interpolate geofacs to both vol/surf nodes, pack back into md
@unpack rxJ = md
rxJ = [Vq;Vf]*rxJ # interp to hybridized points
@pack! md = rxJ

################################################################################
## build global matrices

Nh,Nq = size(VhP)
function rhs(uh,rd,md)
    @unpack rxJ,nxJ,mapP = md
    uP = ((x->x[mapP]) ∘ (x->x[Nq+1:Nh,:]))(uh)
    rhs = rxJ.*(Qrhskew*uh)
    rhs[Nq+1:Nh,:] += .5*uP.*nxJ
    return rhs
end

function rhsB(uh,rd,md)
    @unpack rxJ,nxJ,mapP = md
    uP = ((x->x[mapP]) ∘ (x->x[Nq+1:Nh,:]))(uh)
    rhs = zero.(rxJ.*(Qrhskew*uh))
    rhs[Nq+1:Nh,:] += .5*(uP.*nxJ)
    return rhs
end

Qg = droptol!(build_rhs_matrix(rhs,size(Vh,1),md.K,rd,md),1e-12)
Bg = droptol!(build_rhs_matrix(rhsB,size(Vh,1),md.K,rd,md),1e-12)
Vqg,Vhg,VhPg = (A->kron(I(md.K),A)).((Vq,Vh,VhP))
#Phg = kron(diagm(vec(1 ./ md.J[1,:])),Ph)
Phg = kron(I(md.K),Ph)

QgTr = sparse(transpose(Qg))
global_SBP_ops = (Qg,QgTr,Bg,Vhg,Phg,VhPg,Vqg)

function LF_dissipation(rhoL,rhouL,EL,rhoR,rhouR,ER)
    QL = rhoL,rhouL,EL
    QR = rhoR,rhouR,ER
    # λL = abs.(wavespeed_1D(QL...))
    # λR = abs.(wavespeed_1D(QR...))
    # λavg = .5*(λL+λR)
    λavg = 1.
    return ((uL,uR)->.5*λavg*(uL-uR)).(QL,QR)
end

function rhs_global(Q,global_SBP_ops)
    Qx,QxTr,Bx,Vh,Ph,VhP,Vq = global_SBP_ops
    Qh = u_vfun((x->VhP*x).(v_ufun((x->Vq*x).(vec.(Q))...))...)
    Qhprim = conservative_to_primitive_beta(Qh...)
    rhsQ = hadamard_sum((QxTr,),tuple ∘ euler_fluxes, Qhprim) .-
            hadamard_sum((abs.(Bx),),tuple ∘ LF_dissipation, Qh)
    return (x->-2*Ph*x).(rhsQ)
end

## Jacobian stuff

# define Euler fluxes directly as functions of conservative variables
function F(UL,UR)
    # convert to flux variables
    function UtoQ(U)
        rho,rhou,E = U
        return (rho,rhou./rho,betafun(U...))
    end
    Fx = euler_fluxes(UtoQ(UL)...,UtoQ(UR)...)
    return SVector{length(Fx)}(Fx...)
end

# mappings between conservative and entropy variables and vice versa
dVdU_fun(U::SVector) = ForwardDiff.jacobian(U->SVector(v_ufun(U...)),U)
dUdV_fun(V::SVector) = ForwardDiff.jacobian(V->SVector(u_vfun(V...)),V)
# "slurping" individual arguments for compatibility with FluxDiffUtils.jl
dVdU_fun(U...) = dVdU_fun(SVector(U))
dUdV_fun(V...) = dUdV_fun(SVector(V))

# AD for jacobians
dFx(uL::SVector,uR::SVector) = ForwardDiff.jacobian(uR->F(uL,uR),uR)
dLF(uL::SVector,uR::SVector) =
    ForwardDiff.jacobian(uR->SVector(LF_dissipation(uL...,uR...)),uR)
# inputs are individual arguments, outputs wrapped in a tuple
dFx(rhoL,rhouL,EL,rhoR,rhouR,ER) = tuple(dFx(SVector(rhoL,rhouL,EL),SVector(rhoR,rhouR,ER)))
dLF(rhoL,rhouL,EL,rhoR,rhouR,ER) = tuple(dLF(SVector(rhoL,rhouL,EL),SVector(rhoR,rhouR,ER)))

# compute "parts" of Jacobians
function rhs_global_jac(Q,global_SBP_ops)

    Qx,QxTr,Bx,Vh,Ph,VhP,Vq = global_SBP_ops

    U = SVector(vec.(Q))
    Uq = (x->Vq*x).(U)
    VUh = SVector((x->VhP*x).(v_ufun(Uq...)))
    Qh = u_vfun(VUh...)
    dVdU = banded_matrix_function(dVdU_fun,Uq)
    dUdV = banded_matrix_function(dUdV_fun,VUh)
    Jblocks   = hadamard_jacobian(tuple(Qx),:skew,dFx,Qh) # compute jacobian for entropy conservative part
    JLFblocks = hadamard_jacobian(tuple(abs.(Bx)),:skew,dLF,Qh) # compute jacobian for dissipation term

    # convert FluxDiffUtils.jl output (tuples of matrices) to global matrices
    dVdU,dUdV,Jac,JLF = (A->hcat(vcat.(A...)...)).((dVdU,dUdV,Jblocks,JLFblocks))

    # compute jacobian using chain rule
    return kron(I(Nfields),-2*Ph) * (Jac - JLF) * dUdV * kron(I(Nfields),VhP) * dVdU * kron(I(Nfields),Vq)
end

function tuple_to_arr(X)
    return vcat(X...)
end

function arr_to_tuple(X)
    Length = length(X)÷Nfields
    return ntuple(i->reshape(X,Length,Nfields)[:,i],Nfields)
end

# # basic FD test
# Q = vec.(Q)
# fQ = rhs_global(Q,global_SBP_ops)
# dQ = (x->1e-6*randn(size(x))).(Q)
# QdQ = Q .+ dQ
# fdQ = rhs_global(QdQ,global_SBP_ops)
# dfQ = (fdQ .- fQ) ./ norm.(dQ)
# Jac = rhs_global_jac(Q,global_SBP_ops)
# dfJac = arr_to_tuple(Jac*vcat(dQ...)) ./ norm.(dQ)
# println("basic FD error test: diff = $(norm.(dfJac .- dfQ))")

################################################################################
## Integrating over time

CN = (N+1)^2/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
t0 = 0

# ts = Time_struct()
# @pack! ts = t0,T,dt

Q0 = vec.(Q0)
u0 = tuple_to_arr(Q0)

function f(u)
    Q  = arr_to_tuple(u)
    fQ = rhs_global(Q,global_SBP_ops)
    return tuple_to_arr(fQ)
end

function df(u,δu;adj=false)
    Q = arr_to_tuple(u)
    J = rhs_global_jac(Q,global_SBP_ops)
    if adj
        return transpose(J)*δu
    end
    return J*δu
end

@unpack wJq = md
function η(u)
    Q = arr_to_tuple(u)
    Q = (x->reshape(x,length(rd.r),md.K)).(Q)
    return sum(wJq.*(Vq*Sfun(Q...)))
end

function ∇η(u)
    Q = arr_to_tuple(u) #reshape into tuple of 1D arrays
    Q = (x->reshape(x,length(rd.r),md.K)).(Q) #reshape into tuple of 2D arrays

    #entropy projection step
    VU = v_ufun((x->Vq*x).(Q)...)
    VU = (x->Vq*Pq*x).(VU)
    VU = (x->wJq.*x).(VU)
    VU = (x->transpose(Vq)*x).(VU)
    VU = vec.(VU)
    return tuple_to_arr(VU)
end

#running derv test of df
δu = randn(size(u0))
h0 = 2^(-10)
Nref = 2
h = zeros(Nref+1)
h[1] = h0
for n=1:Nref
    h[n+1] = h[n]/2
end
errs = zeros(Nref+1)
rate = zeros(Nref)

f0 = f(u0)
df0 = df(u0,δu)

dfh_hist = Array[]
df0_hist = Array[]
for n=1:Nref+1
    uh = u0 + h[n].*δu
    fh = f(uh)
    dfh = (fh - f0) / h[n] #./norm(δu)
    push!(dfh_hist,dfh)
    push!(df0_hist,df0)
    errs[n] = norm( dfh - df0 )
end
@show errs

for n=1:Nref
    rate[n] = log2(errs[n]) - log2(errs[n+1])
end
@show rate
#
#
# # arrks = AdjRRK_struct()
# # @pack! arrks = f,df,η,∇η
# # @pack! arrks = u0
# # arrks.u0_lin = u0
# # arrks.return_time = true
# # arrks.return_Δη = true
# #
# # #Running RK
# # RK_solver!(arrks,ts,rk4)
# # Δη_RK = arrks.Δη
# # t_RK = ts.t
# # Q_RK = arr_to_tuple(arrks.u[:,end])
# # Q_RK = (x->reshape(x,length(rd.r),md.K)).(Q_RK)
# # # gr(aspect_ratio=1,legend=false)
# # display(scatter((x->rd.Vp*x).((x,Q_RK[1])),zcolor=rd.Vp*Q_RK[1],msw=0,ms=2,cam=(0,90),legend=false))
# # display(plot(t_RK,Δη_RK))
# #
# # #Running lin RK
# # RK_solver!(arrks,ts,rk4;lin=true)
# # W_RK = arr_to_tuple(arrks.u_lin[:,end])
# # W_RK = (x->reshape(x,length(rd.r),md.K)).(W_RK)
# # display(scatter((x->rd.Vp*x).((x,W_RK[1])),zcolor=rd.Vp*W_RK[1],msw=0,ms=2,cam=(0,90),legend=false))
# #
# #
# #
# #
# #
# # #Running deriv test on lin RK
# # # arrks.return_time = false
# # # arrks.return_Δη = false
# # # arrks.u0_lin = randn(size(u0))
# # # Nref = 10
# # # h0 = 2^(-10)
# # # arrks_h = AdjRRK_struct(arrks)
# # # @pack! arrks_h = f,df
# # # errs,rate,h = AdjRRK.derv_test!(RK_solver!,arrks,arrks_h,ts,rk4,h0,Nref)
# #
# # # #Running RRK
# # # RRK_solver!(arrks,ts,rk4)
# # # Δη_RRK = arrks.Δη
# # # γ_RRK  = arrks.γ
# # # t_RRK  = ts.t
# # # Q_RRK = arr_to_tuple(arrks.u[:,end])
# # # Q_RRK = (x->reshape(x,length(rd.r),md.K)).(Q_RRK)
# # # display(scatter((x->rd.Vp*x).((x,Q_RRK[1])),zcolor=rd.Vp*Q_RRK[1],msw=0,ms=2,cam=(0,90),legend=false))
# # # display(plot(t_RRK,Δη_RRK))
# # # display(plot(t_RRK,γ_RRK))
