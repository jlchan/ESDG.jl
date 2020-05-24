push!(LOAD_PATH, "./src")

# "Packages"
using Revise # reduce need for recompile
using Plots
using Documenter
using LinearAlgebra
using SparseArrays

# "User defined modules"
using CommonUtils
using Basis1D
using Basis2DQuad
using Basis3DHex
using UniformHexMesh

using SetupDG
using UnPack

N = 3
K1D = 8
CFL = .5
T = 1 # endtime

VX,VY,VZ,EToV = uniform_hex_mesh(2*K1D,K1D,K1D)

rd = init_reference_hex(N,gauss_quad(0,0,N))
md = init_mesh((VX,VY,VZ),EToV,rd)

# make domain periodic
@unpack Nfaces = rd
@unpack xf,yf,zf,K,mapM,mapP,mapB = md
LX = 2; LY = 2; LZ = 2
mapPB = build_periodic_boundary_maps(xf,yf,zf,LX,LY,LZ,
                                    Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)*3/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

function rhs(u,rd::RefElemData,md::MeshData)

    @unpack Dr,Ds,Dt,LIFT,Vf = rd
    @unpack rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J = md
    @unpack nxJ,nyJ,nzJ,sJ = md
    @unpack mapP,mapB = md

    uf = Vf*u
    du = uf[mapP]-uf
    uflux = @. .5*(du*nxJ - 0*abs(nxJ)*du)

    ur,us,ut = (A->A*u).((Dr,Ds,Dt))
    uxJ      = @. rxJ*ur + sxJ*us + txJ*ut
    rhsu     = uxJ + LIFT*uflux

    return @. -rhsu/J
end

# set initial conditions
@unpack x,y,z = md
u = @. exp(-25*(x^2+y^2))

resQ = zeros(size(x))
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ    = rhs(u,rd,md)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. u   += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps: $i out of $Nsteps")
    end
end

# "plotting nodes"
gr(aspect_ratio=1,legend=false,
   markerstrokewidth=0,markersize=2)

@unpack Vp = rd
xp,yp,zp = (x->Vp*x).((x,y,z))
vv = Vp*u
ids = map(x->x[1],findall(@. abs(zp[:]-1)<1e-10))
(xp,yp,zp,vv) = (x->x[ids]).((xp,yp,zp,vv))
scatter(xp,yp,vv,zcolor=vv,camera=(0,90))
