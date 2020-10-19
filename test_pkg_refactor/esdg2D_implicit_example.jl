using BenchmarkTools
using LoopVectorization
using LinearAlgebra
using Plots
using UnPack

using EntropyStableEuler.Fluxes2D
import EntropyStableEuler: γ

using FluxDiffUtils
using StartUpDG
using StartUpDG.ExplicitTimestepUtils

include("HybridizedSBPUtils.jl")
using .HybridizedSBPUtils

N = 2
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

Ax,Ay,Bx,By,B = assemble_global_SBP_matrices_2D(rd, md, Qrhskew, Qshskew)

function euler_fluxes_conservative_vars(UL,UR)
    QL = conservative_to_primitive_beta(UL[1],UL[2],UL[3],UL[4])
    QR = conservative_to_primitive_beta(UR[1],UR[2],UR[3],UR[4])
    return SVector{4}(euler_fluxes(QL[1],QL[2],QL[3],QL[4],
                                   QR[1],QR[2],QR[3],QR[4]))
end

#
# jac = hadamard_jacobian((Ax,Ay),:sym,dF,Q)
