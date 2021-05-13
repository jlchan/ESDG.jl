using Plots
using UnPack

using StartUpDG
using StartUpDG.ExplicitTimestepUtils

N = 3
K1D = 16
FinalTime = .5
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
u0(x,y) = @. sin(pi*x)*sin(pi*y)
u = u0(x,y)

# Evaluates the semi-discrete RHS for du/dt + du/dx = 0
function rhs(u, rd::RefElemData, md::MeshData)
    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    uf = Vf*u
    du = @. uf[mapP]-uf
    tau = .5
    uflux = @. .5*du*nxJ - tau*abs(nxJ)*du

    ur = Dr*u
    us = Ds*u
    dudx = @. rxJ*ur + sxJ*us
    rhsu = dudx + LIFT*uflux
    return @. -rhsu/J
end

#######################################################
##### Perform time-stepping
#######################################################

# Runge-Kutta time integration coefficients
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # estimated trace constant
hmin = 2/K1D # estimate based on uniform mesh
dt = CFL * hmin / (CN)
Nsteps = ceil(Int,FinalTime/dt)
dt = FinalTime/Nsteps # ensure exactly Nsteps

resQ = zero(u)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsu = rhs(u,rd,md)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsu
        @. u += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
    end
end

# plot solution at equally spaced nodes on each element
gr(aspect_ratio=1,legend=false)
@unpack Vp = rd # interpolate from interpolation nodes to plotting nodes
scatter((x->Vp*x).((x,y,u)),zcolor=Vp*u, msw=0,ms=2,cam=(0,90))

# compute L2 error using quadrature
@unpack Vq = rd
@unpack xq,yq,wJq = md
err = sqrt(sum(wJq.*(u0(xq .- FinalTime,yq) - Vq*u).^2))
plot!(title="L2 error = $err")
