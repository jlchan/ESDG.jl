using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh
using Setup2DTri
using UnPack

"Define approximation parameters"
N   = 3 # The order of approximation
K1D = 16 # number of elements along each edge of a rectangle
CFL = .5 # relative size of a time-step
T   = 1.5 # final time

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
# mapP[mapB] = mapPB
# @pack! md = mapP

"=========== Done defining geometry and mesh ============="

"Time integration coefficients"
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Define the initial conditions by interpolation"
p = @. exp(-100*(x^2+y^2))
pprev = copy(p) # 1st order accurate approximation to dp/dt = 0

"Define function to evaluate the RHS"
function rhs_2ndorder(p,rd::RefElemData,md::MeshData)
    # unpack arguments
    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = md
    @unpack mapP,mapB = md

    # construct sigma
    pf = Vf*p # eval pressure at face points

    # imposing zero Dirichlet BCs
    pP = pf[mapP]
    pP[mapB] = -pf[mapB]
    dp = pP-pf # compute jumps of pressure

    pr = Dr*p
    ps = Ds*p
    dpdx = @. rxJ*pr + sxJ*ps
    dpdy = @. ryJ*pr + syJ*ps
    σxflux = @. dp*nxJ
    σyflux = @. dp*nyJ
    σx = (dpdx + .5*LIFT*σxflux)./J
    σy = (dpdy + .5*LIFT*σyflux)./J

    # compute div(σ)
    σxf,σyf = (x->Vf*x).((σx,σy))
    σxP,σyP = (x->x[mapP]).((σxf,σyf))
    pflux = @. .5*((σxP-σxf)*nxJ + (σyP-σyf)*nyJ)
    σxr,σyr = (x->Dr*x).((σx,σy))
    σxs,σys = (x->Ds*x).((σx,σy))
    dσxdx = @. rxJ*σxr + sxJ*σxs
    dσydy = @. ryJ*σyr + syJ*σys

    tau = 1/2
    rhsp = dσxdx + dσydy + LIFT*(pflux + tau*dp)

    return rhsp./J
end

#plotting nodes
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0,markersize=2,
   camera=(0,90),#zlims=(-1,1),clims=(-1,1),
   axis=nothing,border=:none)

# Perform time-stepping
for i = 2:Nsteps

    rhsQ = rhs_2ndorder(p,rd,md)
    pnew = 2*p - pprev + dt^2 * rhsQ
    @. pprev = p
    @. p = pnew

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        # vv = Vp*p
        # scatter(Vp*x,Vp*y,vv,zcolor=vv)
    end
end

@unpack Vp = rd
vv = Vp*p
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
#@show maximum(abs.(p-pex(x,y,T)))
