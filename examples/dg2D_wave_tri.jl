using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using Basis2DTri
using UniformTriMesh

using SetupDG
using UnPack

"Define physical and discretization parameters"
N   = 3 # The order of approximation
K1D = 16 # number of elements along each edge of a rectangle
CFL = 1 # relative size of a time-step
T   = 4 # final time

"=========== Setup reference element and mesh ============="

# construct mesh
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)

# intialize reference operators
rd = init_reference_tri(N)

# initialize physical mesh data
md = init_mesh((VX,VY),EToV,rd)

#Make boundary maps periodic
@unpack Nfaces,Vf = rd
@unpack x,y,K,mapM,mapP,mapB = md
xf,yf = (x->Vf*x).((x,y))
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

"======== Define initial coefficients and time-stepping =========="

"Define the initial conditions by interpolation"
p = @. exp(-10*(x^2+y^2))
u = @. 0*x
v = @. 0*x

"Define function to evaluate the RHS: Q = (p,u,v)"
function rhs(Q, rd::RefElemData, md::MeshData)

    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J = md
    @unpack nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    Qf = (x->Vf*x).(Q)
    QP = (x->x[mapP]).(Qf)
    dp,du,dv = QP.-Qf

    tau = .5
    dun = @. (du*nxJ + dv*nyJ)/sJ
    pflux = @. .5*(du*nxJ + dv*nyJ) - tau*dp*sJ
    uflux = @. .5*dp*nxJ - tau*dun*nxJ
    vflux = @. .5*dp*nyJ - tau*dun*nyJ

    pr,ur,vr = (x->Dr*x).(Q)
    ps,us,vs = (x->Ds*x).(Q)
    dudx = @. rxJ*ur + sxJ*us
    dvdy = @. ryJ*vr + syJ*vs
    dpdx = @. rxJ*pr + sxJ*ps
    dpdy = @. ryJ*pr + syJ*ps
    rhsp = dudx + dvdy + LIFT*pflux
    rhsu = dpdx        + LIFT*uflux
    rhsv = dpdy        + LIFT*vflux

    return (x->-x./J).((rhsp,rhsu,rhsv))
end

"Time integration coefficients"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Perform time-stepping"
Q = [p,u,v] # use mutable arrays for Q, resQ for scoping
resQ = [zeros(size(x)) for i in eachindex(Q)]
for i = 1:Nsteps
    for INTRK = 1:5
        time = i*dt + rk4c[INTRK]*dt
        rhsQ = rhs(Q,rd,md)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q    = Q + rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

# plotting nodes
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0,markersize=2,
   camera=(0,90))

@unpack Vp = rd
vv = Vp*Q[1]
scatter(Vp*x,Vp*y,vv,zcolor=vv)
