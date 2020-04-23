using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using UnPack
using SparseArrays

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
# using Basis2DTri
# using UniformTriMesh
# using Setup2DTri
using Basis1D
using Basis2DQuad
using UniformQuadMesh
using Setup2DQuad

"Define approximation parameters"
N   = 4 # The order of approximation
K1D = 8 # number of elements along each edge of a rectangle
CFL = .25 # relative size of a time-step
T   = .1 # final time

"=========== Setup code ============="

# construct mesh
VX,VY,EToV = uniform_quad_mesh(K1D,K1D)

iids = findall(@. (abs(abs(VX)-1) > 1e-12) & (abs(abs(VY)-1) > 1e-12))
@. VX[iids] += .1/K1D*randn()
@. VY[iids] += .1/K1D*randn()

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

# redefine gauss lobatto quadrature
r1D,w1D = gauss_lobatto_quad(0,0,N)
e = ones(size(r1D))
rf = [r1D; e; -r1D; -e]
sf = [-e; r1D; e; -r1D]
wf = vec(repeat(w1D,Nfaces,1));
Vf = vandermonde_2D(N,rf,sf)/V
LIFT = M\(Vf'*diagm(wf))
Pq = M\(Vq'*diagm(wq))
@pack! rd = Vf,Vq,Pq,LIFT

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
    @unpack K,mapP,mapB = md

    # split form volume terms
    # ur,us  = (A->A*u).((Dr,Ds))
    # dudx   = @. rxJ*ur + sxJ*us
    # dudy   = @. ryJ*ur + syJ*us
    f_proj = (u.^2)
    #
    # du2x  = rxJ.*(Dr*f_proj) + sxJ.*(Ds*f_proj)
    # du2y  = ryJ.*(Dr*f_proj) + syJ.*(Ds*f_proj)
    dudx,dudy,du2x,du2y = ntuple(x->zeros(size(u)),4)
    for e = 1:K
        Dx = .5*(diagm(rxJ[:,e])*Dr + diagm(sxJ[:,e])*Ds
            + Dr*diagm(rxJ[:,e]) + Ds*diagm(sxJ[:,e]))
        Dy = .5*(diagm(ryJ[:,e])*Dr + diagm(syJ[:,e])*Ds
            + Dr*diagm(ryJ[:,e]) + Ds*diagm(syJ[:,e]))
        dudx[:,e] = Dx*u[:,e]
        dudy[:,e] = Dy*u[:,e]
        du2x[:,e] = Dx*f_proj[:,e]
        du2y[:,e] = Dy*f_proj[:,e]
    end
    du2   = du2x + du2y # (du^2/dx + du^2/dy, v)
    udu   = u.*(dudx+dudy) # (u*dudx,v)

    # boundary terms
    uf = Vf*u
    uP = uf[mapP]
    du = uP - uf
    fproj_f = Vf*f_proj

    tau = 0
    uflux = @. ((1/6)*(uP^2 + uP*uf) - (1/3)*fproj_f)*(nxJ + nyJ) - .5*tau*du*max(abs(uP),abs(uf))*abs(nxJ + nyJ)

    rhsu = (1/3)*(du2 + udu) + LIFT*uflux

    # rhsu = dudx + LIFT*(@. .5 * du * nxJ)

    return -rhsu./J
end

# plotting nodes
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=2)
plot()

@unpack Vp = rd
xp,yp = (x->Vp*x).((x,y))

@unpack J = md

"Perform time-stepping"
resQ = zeros(size(x))
rhstest = zeros(Nsteps)
interval = 5
for i = 1:Nsteps
    for INTRK = 1:5
        time    = i*dt + rk4c[INTRK]*dt
        rhsQ    = rhs(u,rd,md)

        if INTRK==5
            rhstest[i] = sum(u.*(J.*(M*rhsQ)))
        end

        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. u    = u + rk4b[INTRK]*resQ
    end

    if i%interval==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

vv = Vp*u
# scatter(xp,yp,vv,zcolor=vv,camera=(3,25))
display(scatter(xp,yp,vv,zcolor=vv,camera=(0,90)))

@show maximum(abs.(rhstest))

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
L2err = sqrt(sum(wJq2.*(Vq2*u - burgers_exact_sol_2D(u0,xq2,yq2,T,dt/250)).^2))
@show L2err
