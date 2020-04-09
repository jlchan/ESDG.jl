using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using Basis2DTri
using UniformTriMesh

using Setup2DTri
using UnPack

"Define approximation parameters"
N   = 3 # The order of approximation
K1D = 32 # number of elements along each edge of a rectangle
CFL = .75 # relative size of a time-step
T   = 1.25 # final time

"=========== Setup code ============="

# construct mesh
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)

# intialize reference operators
rd = init_reference_tri(N)

# initialize physical mesh data
md = init_tri_mesh((VX,VY),EToV,rd)

#Make boundary maps periodic
@unpack Nfaces,Vf = rd
@unpack x,y,K,mapM,mapP,mapB = md
xf,yf = (x->Vf*x).((x,y))
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
# @pack! md = mapP

"======== Define initial coefficients and time-stepping =========="

"Define the initial conditions by interpolation"
u = @. -sin(pi*x)*sin(pi*y)
# u = @. exp(-10*(x^2+y^2))
# ufun(x,y) = @. ((x<0) & (y > 0))*(-.2) + ((x>0) & (y > 0))*(-1) + ((x<0) & (y < 0))*(.5) + ((x>0) & (y < 0))*(.8)
# @unpack Vq,Pq = rd
# xq,yq = (x->Vq*x).((x,y))
# u = Pq*ufun(xq,yq)


"Time integration coefficients"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Define function to evaluate the RHS: Q = (p,u,v)"
function rhs(u, rd::RefElemData, md::MeshData, uBC)

    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J = md
    @unpack nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    # quadrature operators
    @unpack Vq,Pq = rd

    # split form volume terms
    uq     = Vq*u
    ur,us  = (A->A*u).((Dr,Ds))
    dudx   = @. rxJ*ur + sxJ*us
    dudy   = @. ryJ*ur + syJ*us
    f_proj = Pq*(uq.^2)


    udu  = Pq*(uq.*(Vq*(dudx+dudy)))
    du2x  = rxJ.*(Dr*f_proj) + sxJ.*(Ds*f_proj)
    du2y  = ryJ.*(Dr*f_proj) + syJ.*(Ds*f_proj)
    du2 = du2x + du2y

    # boundary terms
    uf = Vf*u
    uP = uf[mapP]
    du = uP - uf

    tau = 1
    fproj_f = Vf*f_proj
    uflux = @. ((1/6)*(uP^2 + uP*uf) - (1/3)*fproj_f)*(nxJ+nyJ) - .5*tau*du*max(abs(uP),abs(uf))*abs(nxJ+nyJ)

    rhsu = (1/3)*(du2 + udu) + LIFT*uflux

    return -rhsu./J
end

# plotting nodes
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0,markersize=2)

@unpack Vp = rd
xp,yp = (x->Vp*x).((x,y))

"Perform time-stepping"
uBC = (Vf*u)[mapB] # copy for BCs
resQ = zeros(size(x))
interval = 5
@gif for i = 1:Nsteps
    for INTRK = 1:5
        time    = i*dt + rk4c[INTRK]*dt
        rhsQ    = rhs(u,rd,md,uBC)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. u    = u + rk4b[INTRK]*resQ
    end

    if i%interval==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        vv = Vp*u
        # scatter(xp,yp,vv,zcolor=vv,camera=(3,25))
        scatter(xp,yp,vv,zcolor=vv,camera=(0,90),border=:none,axis=nothing)
    end
end every interval
