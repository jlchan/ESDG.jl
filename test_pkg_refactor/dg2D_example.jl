using LoopVectorization
using Plots
using UnPack

using StartUpDG
using StartUpDG.ExplicitTimestepUtils

N = 3
K1D = 16
T = .5
CFL = 1.

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
p = @. exp(-100*(x^2+y^2))
u = @. 0*x
v = @. 0*x

# "Define function to evaluate the RHS: Q = (p,u,v)"
function rhs(Q, rd::RefElemData, md::MeshData)

    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J = md
    @unpack nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    Qf = (x->Vf*x).(Q)
    QP = (x->x[mapP]).(Qf)
    dp,du,dv = QP.-Qf

    tau = .5
    dun   = @. (du*nxJ + dv*nyJ)/sJ
    pflux = @. .5*(du*nxJ + dv*nyJ) - tau*dp*sJ
    uflux = @. .5*dp*nxJ - tau*dun*nxJ
    vflux = @. .5*dp*nyJ - tau*dun*nyJ

    pr,ur,vr = (x->Dr*x).(Q)
    ps,us,vs = (x->Ds*x).(Q)
    dpdx = @. rxJ*pr + sxJ*ps
    dpdy = @. ryJ*pr + syJ*ps
    dudx = @. rxJ*ur + sxJ*us
    dvdy = @. ryJ*vr + syJ*vs
    rhsp = dudx + dvdy + LIFT*pflux
    rhsu = dpdx        + LIFT*uflux
    rhsv = dpdy        + LIFT*vflux

    return map(x->-x./J,(rhsp,rhsu,rhsv))
end

function rhs_avx(Q, rd::RefElemData, md::MeshData)

    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J = md
    @unpack nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    Qf = (x->Vf*x).(Q)
    QP = (x->x[mapP]).(Qf)
    dp,du,dv = QP.-Qf

    tau = .5
    dun   = @avx @. (du*nxJ + dv*nyJ)/sJ
    pflux = @avx @. .5*(du*nxJ + dv*nyJ) - tau*dp*sJ
    uflux = @avx @. .5*dp*nxJ - tau*dun*nxJ
    vflux = @avx @. .5*dp*nyJ - tau*dun*nyJ

    pr,ur,vr = (x->Dr*x).(Q)
    ps,us,vs = (x->Ds*x).(Q)
    dpdx = @avx @. rxJ*pr + sxJ*ps
    dpdy = @avx @. ryJ*pr + syJ*ps
    dudx = @avx @. rxJ*ur + sxJ*us
    dvdy = @avx @. ryJ*vr + syJ*vs
    rhsp = dudx + dvdy + LIFT*pflux
    rhsu = dpdx        + LIFT*uflux
    rhsv = dpdy        + LIFT*vflux

    return (x->@avx -x./J).((rhsp,rhsu,rhsv))
end

# "Time integration coefficients"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# "Perform time-stepping"
Q = (p,u,v) # use mutable arrays for Q, resQ for scoping
resQ = zero.(Q)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs_avx(Q,rd,md)
        bcopy!.(resQ, @. rk4a[INTRK]*resQ + dt*rhsQ)
        bcopy!.(Q,@. Q + rk4b[INTRK]*resQ)
    end

    if i%10==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
    end
end

# # dopri initialization
# rka,rkE,rkc = dp56()
# PIparams = init_PI_controller(5)
# Qtmp = similar.(Q)
# rhsQrk = ntuple(x->zero.(Q),length(rkE))
# prevErrEst = nothing
# rhsQ = rhs(Q,rd,md)
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
#     global t,dt,i,prevErrEst # modify compute_adaptive_dt!(...prevErrEst)
#     accept_step, dtnew, prevErrEst = compute_adaptive_dt(
#                                         Q,rhsQrk,dt,rkE,PIparams,prevErrEst)
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

# plotting nodes
gr(aspect_ratio=1,legend=false)
using NodesAndModes.Tri
rp,sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/rd.VDM
scatter((x->Vp*x).((x,y,Q[1])),zcolor=Vp*Q[1], msw=0,ms=2,cam=(0,90))
