using Plots
using UnPack

using StartUpDG
using StartUpDG.ExplicitTimestepUtils

N = 3
K1D = 16
T = .5
CFL = .5

# init ref element and mesh
rd = init_reference_tri(N)
VX,VY,EToV = uniform_tri_mesh(K1D)

# # uncomment these lines to switch to quad meshes
# rd = init_reference_quad(N)
# VX,VY,EToV = uniform_quad_mesh(K1D)

# construct DG mesh and node connectivities
md = init_DG_mesh(VX,VY,EToV,rd)

# Make domain periodic
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
build_periodic_boundary_maps!(md,rd,LX,LY)

#######################################################
##### Define initial coefficients and rhs
#######################################################

@unpack x,y = md
p = @. exp(-100*(x^2+y^2))
u = @. 0*x
v = @. 0*x

# "Define function to evaluate the RHS: Q = (p,u,v)"
function rhs(Q, rd::RefElemData, md::MeshData)

    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = md
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

    return map(x->(@. -x/J),(rhsp,rhsu,rhsv))
end

#######################################################
##### Perform time-stepping
#######################################################

# Runge-Kutta time integration coefficients
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
hmin = 2/K1D # estimate based on uniform mesh
dt = CFL * hmin / (CN)
Nsteps = ceil(Int,T/dt)
dt = T/Nsteps # ensure exactly Nsteps

Q = (p,u,v) # use mutable arrays for Q
resQ = zero.(Q)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,rd,md)
        bcopy!.(resQ, @. rk4a[INTRK]*resQ + dt*rhsQ)
        bcopy!.(Q,@. Q + rk4b[INTRK]*resQ)
    end

    if i%10==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
    end
end

# plot solution at equally spaced nodes on each element
gr(aspect_ratio=1,legend=false)
@unpack Vp = rd # interpolate from interpolation nodes to plotting nodes
scatter((x->Vp*x).((x,y,Q[1])),zcolor=Vp*Q[1], msw=0,ms=2,cam=(0,90))
