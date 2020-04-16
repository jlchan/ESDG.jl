using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using UnPack

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
# using Basis2DTri
# using UniformTriMesh
# using Setup2DTri
using Basis2DQuad
using UniformQuadMesh
using Setup2DQuad

"Define approximation parameters"
N   = 5 # The order of approximation
K1D = 8 # number of elements along each edge of a rectangle
CFL = .75 # relative size of a time-step
T   = .1 # final time

"=========== Setup code ============="

# construct mesh
VX,VY,EToV = uniform_quad_mesh(K1D,K1D)

# intialize reference operators
rd = init_reference_quad(N)

# initialize physical mesh data
md = init_quad_mesh(VX,VY,EToV,rd)

#Make boundary maps periodic
@unpack Nfaces,Vf = rd
@unpack x,y,K,mapM,mapP,mapB = md
xf,yf = (x->Vf*x).((x,y))
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
# @pack! md = mapP

"======== Modify quadrature =========="

@unpack r,s,V,Vq,wq,Vf,wf = rd
M = transpose(Vq)*diagm(wq)*Vq

# redefine quadrature operators
rq,sq = (r,s)
wq = vec(sum(M,dims=2)) # apply mass lumping to get integrals
Vq = vandermonde_2D(N,rq,sq)/V
M  = transpose(Vq)*diagm(wq)*Vq
LIFT = M\(Vf'*diagm(wf))
Pq = M\(Vq'*diagm(wq))
@pack! rd = Vq,Pq,LIFT

"======== Define initial coefficients and time-stepping =========="

"Define the initial conditions by interpolation"
u0(x,y) = @. -sin(pi*x)*sin(pi*y)
u = u0(x,y)

"Time integration coefficients"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps


"Define function to evaluate the RHS: Q = (p,u,v)"
function rhs(u, rd::RefElemData, md::MeshData)

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

    du2x  = rxJ.*(Dr*f_proj) + sxJ.*(Ds*f_proj)
    du2y  = ryJ.*(Dr*f_proj) + syJ.*(Ds*f_proj)
    du2   = du2x + du2y # (du^2/dx + du^2/dy, v)
    udu   = Pq*(uq.*(Vq*(dudx+dudy))) # (u*dudx,v)

    # boundary terms
    uf = Vf*u
    uP = uf[mapP]
    du = uP - uf
    fproj_f = Vf*f_proj

    tau = 1
    uflux = @. ((1/6)*(uP^2 + uP*uf) - (1/3)*fproj_f)*(nxJ + nyJ) - .5*tau*du*max(abs(uP),abs(uf))*abs(nxJ + nyJ)

    rhsu = (1/3)*(du2 + udu) + LIFT*uflux

    return -rhsu./J
end

# plotting nodes
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=2)
plot()

@unpack Vp = rd
xp,yp = (x->Vp*x).((x,y))

"Perform time-stepping"
resQ = zeros(size(x))
interval = 5
for i = 1:Nsteps
    for INTRK = 1:5
        time    = i*dt + rk4c[INTRK]*dt
        rhsQ    = rhs(u,rd,md)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. u    = u + rk4b[INTRK]*resQ
    end

    if i%interval==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

# vv = Vp*u
# scatter(xp,yp,vv,zcolor=vv,camera=(3,25))
# scatter(xp,yp,vv,zcolor=vv,camera=(0,90))


function burgers_exact_sol_2D(u0,x,y,T,dt)
    Nsteps = ceil(Int,T/dt)
    dt = T/Nsteps
    u = u0(x,y) # computed at input points
    for i = 1:Nsteps
        t = i*dt
        u .= @. u0(x-u*t,y-u*t) # evolve solution at quadrature points using characteristics
    end
    return u
end

@unpack J = md
rq2,sq2,wq2 = quad_nodes_2D(3*N)
Vq2 = vandermonde_2D(N,rq2,sq2)/V
xq2,yq2 = (x->Vq2*x).((x,y))
wJq2 = diagm(wq2)*(Vq2*J)
L2err = sqrt(sum(wJq2.*(Vq2*u - burgers_exact_sol_2D(u0,xq2,yq2,T,dt/100)).^2))
@show L2err
