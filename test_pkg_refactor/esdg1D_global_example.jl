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

include("HybridizedSBPUtils.jl")
using .HybridizedSBPUtils

N = 3
K1D = 32
CFL = .25
T = .5

rd = init_reference_interval(N)
VX,EToV = uniform_1D_mesh(K1D)
md = init_DG_mesh(VX,EToV,rd)

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
rho = @. 1 + .25*exp(-100*((x-.1)^2))
u = @. 0*x + .0
p = @. rho^γ

Q = primitive_to_conservative(rho,u,p)

# interpolate geofacs to both vol/surf nodes, pack back into md
@unpack Vq,Vf = rd
@unpack rxJ = md
rxJ = [Vq;Vf]*rxJ # interp to hybridized points
@pack! md = rxJ

#######################################################
#               build global matrices                 #
#######################################################

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
    rhs = zero.(uh)
    rhs[Nq+1:Nh,:] += .5*(uP.*nxJ)
    return rhs
end

Qg = droptol!(build_rhs_matrix(rhs,size(Vh,1),md.K,rd,md),1e-12)
Bg = droptol!(build_rhs_matrix(rhsB,size(Vh,1),md.K,rd,md),1e-12)
Vqg,Vhg,VhPg = (A->kron(I(md.K),sparse(A))).((Vq,Vh,VhP))
Phg = kron(diagm(vec(1 ./ md.J[1,:])),Ph)

global_SBP_ops = (Qg,sparse(Qg'),Bg,Vhg,Phg,VhPg,Vqg)

function LF_dissipation(rhoL,rhouL,EL,rhoR,rhouR,ER)
    QL = rhoL,rhouL,EL
    QR = rhoR,rhouR,ER
    λL = abs.(wavespeed_1D(QL...))
    λR = abs.(wavespeed_1D(QR...))
    return ((uL,uR)->.5*max.(λL,λR)*(uR-uL)).(QL,QR)
end

function rhs_global(Q,global_SBP_ops)
    Qx,QxTr,B,Vh,Ph,VhP,Vq = global_SBP_ops
    Qh = u_vfun((x->VhP*x).(v_ufun((x->Vq*x).(vec.(Q))...))...)
    Qhprim = conservative_to_primitive_beta(Qh...)
    rhsQ = hadamard_sum((QxTr,),tuple ∘ euler_fluxes, Qhprim) .-
            hadamard_sum((abs.(B),),tuple ∘ LF_dissipation, Qh)
    return (x->-2*Ph*x).(rhsQ)
end

## jacobian computations

# define Euler fluxes directly as functions of conservative variables.
function F(UL,UR)
    # convert to flux variables
    function UtoQ(U)
        rho,rhou,E = U
        beta = betafun(U...)
        return (rho,rhou./rho,beta)
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
dLF(uL::SVector,uR::SVector) = ForwardDiff.jacobian(
                uR->SVector(LF_dissipation(uL...,uR...)),uR)
# inputs are individual arguments, outputs wrapped in a tuple
dFx(rhoL,rhouL,EL,rhoR,rhouR,ER) = tuple(dFx(SVector(rhoL,rhouL,EL),SVector(rhoR,rhouR,ER)))
dLF(rhoL,rhouL,EL,rhoR,rhouR,ER) = tuple(dLF(SVector(rhoL,rhouL,EL),SVector(rhoR,rhouR,ER)))

# compute "parts" of Jacobians
U = SVector(vec.(Q))
Uq = (x->Vqg*x).(U)
VUh = SVector((x->VhPg*x).(v_ufun(Uq...)))
Qh = u_vfun(VUh...)
dVdU = banded_matrix_function(dVdU_fun,Uq)
dUdV = banded_matrix_function(dUdV_fun,VUh)
Jblocks = hadamard_jacobian(tuple(Qg),:skew,dFx,Qh) # compute jacobian for entropy conservative part
JLFblocks = hadamard_jacobian(tuple(abs.(Bg)),:skew,dLF,Qh) # compute jacobian for dissipation term

# convert FluxDiffUtils.jl output (tuples of matrices) to global matrices
dVdU,dUdV,Jac,JLF = (A->hcat(vcat.(A...)...)).((dVdU,dUdV,Jblocks,JLFblocks))

# compute jacobian using chain rule
Nfields = 3
dfdu = kron(I(Nfields),-2*Phg) * (Jac + JLF) * dUdV * kron(I(Nfields),VhPg) * dVdU * kron(I(Nfields),Vqg)

h = 1e-5
dQ = ntuple(x->h*randn(size(first(Q))),Nfields)
fQ = rhs_global(Q,global_SBP_ops)
fdQ = rhs_global(Q .+ dQ, global_SBP_ops)
df = vcat(((fdQ .- fQ))...)
df2 = dfdu * vcat(vec.(dQ)...)
@show norm(df-df2)

#######################################################
#               local rhs computations                #
#######################################################

# "Time integration coefficients"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)^2/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# "Perform time-stepping"
Q = vec.(Q)
resQ = zero.(Q)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs_global(Q,global_SBP_ops)
        bcopy!.(resQ, @. rk4a[INTRK]*resQ + dt*rhsQ)
        bcopy!.(Q,@. Q + rk4b[INTRK]*resQ)
    end

    if i%10==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
    end
end

# plotting nodes
Q = (x->reshape(x,length(rd.r),md.K)).(Q)
gr(aspect_ratio=1,legend=false)
scatter((x->rd.Vp*x).((x,Q[1])),zcolor=rd.Vp*Q[1],msw=0,ms=2,cam=(0,90))
