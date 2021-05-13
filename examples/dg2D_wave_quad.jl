using Revise # reduce need for recompile
using Plots
using Documenter
using LinearAlgebra
using SparseArrays

# "User defined modules"
push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

using SetupDG
using UnPack

"Approximation parameters"
N   = 3  # order of approximation
K1D = 16 # number of elements per side
CFL = .75
T   = .75

"======= initialize reference element data and mesh ======="

# specify lobatto nodes to automatically get DG-SEM mass lumping
rd = init_reference_quad(N,gauss_lobatto_quad(0,0,N))

VX, VY, EToV = uniform_quad_mesh(K1D, K1D)
md = init_mesh((VX,VY),EToV,rd)

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

" ====== set up initial conditions ====== "

@unpack x,y = md
p = @. exp(-100*(x^2+(y-.25)^2))
u = zeros(size(x))
v = zeros(size(x))

"Time integration"
rk4a, rk4b, rk4c = rk45_coeffs()
CN = (N+1)*(N+2)  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"varying wavespeed"
c2 = @. 1 + .5*sin(pi*x)*sin(pi*y)
# c2 = ones(size(x))

function rhs(Q, rd::RefElemData, md::MeshData, params...)

    (p,u,v) = Q
    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    QM = (x->Vf*x).(Q)
    QP = (x->x[mapP]).(QM)
    (dp,du,dv) = QP.-QM

    "compute central numerical flux"
    # (pM,uM,vM) = QM
    # du[mapB] = -2*uM[mapB]
    # dv[mapB] = -2*vM[mapB]
    pflux = @. du*nxJ + dv*nyJ
    uflux = @. dp*nxJ
    vflux = @. dp*nyJ

    pr,ur,vr = (x->Dr*x).(Q)
    ps,us,vs = (x->Ds*x).(Q)

    px = @. rxJ*pr + sxJ*ps;
    py = @. ryJ*pr + syJ*ps
    ux = @. rxJ*ur + sxJ*us;
    vy = @. ryJ*vr + syJ*vs

    rhsp = (ux+vy) + .5*LIFT*pflux
    rhsu = px + .5*LIFT*uflux
    rhsv = py + .5*LIFT*vflux

    c2 = params[1]
    rhsp = @. c2*rhsp
    return (x->-x./J).((rhsp,rhsu,rhsv))
end

Q = [p,u,v] # make arrays of arrays for mutability
resQ = [zeros(size(x)) for i in eachindex(Q)]
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,rd,md,c2)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q    = Q + rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps: $i out of $Nsteps")
    end
end

#plotting
gr(aspect_ratio=1,legend=false,
   markerstrokewidth=0,markersize=2)

@unpack Vp = rd # interpolation nodes
vv = Vp*Q[1]
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
