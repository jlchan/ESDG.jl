using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using SparseArrays

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 3 # The order of approximation
K   = 32
CFL = .5
T   = 2.25

# viscosity, wave speed
ϵ   = .0001
a   = 1

"Mesh related variables"
VX = LinRange(-1,1,K+1)
EToV = repeat([0 1],K,1) + repeat(1:K,1,2)

"Construct matrices on reference elements"
r,w = gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, r)
Dr = grad_vandermonde_1D(N, r)/V
M = inv(V*V')

"Nodes on faces, and face node coordinate"
wf = [1;1]
Vf = vandermonde_1D(N,[-1;1])/V
LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix

"Construct global coordinates"
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
x = V1*VX[transpose(EToV)]

"Connectivity maps"
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

"Make maps periodic"
mapP[1] = mapM[end]
mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1

"=========== done with mesh setup here ============ "

"pack arguments into tuples"
ops = (Dr,LIFT,Vf)
vgeo = (rxJ,J)
fgeo = (nxJ,)

function rhs(u,ops,vgeo,fgeo,mapP,params...)
    # unpack arguments
    Dr,LIFT,Vf = ops
    rxJ,J = vgeo
    nxJ, = fgeo

    # construct sigma
    uf = Vf*u
    du = uf[mapP]-uf
    σxflux = @. .5*du*nxJ
    dudx = rxJ.*(Dr*u)
    σx = (dudx + LIFT*σxflux)./J

    # define viscosity, wavespeed parameters
    ϵ = params[1]
    a = params[2]
    tau = 1

    # compute dσ/dx
    σxf = Vf*σx
    σxP = σxf[mapP]
    σflux = @. .5*((σxP-σxf)*nxJ + tau*du)
    dσxdx = rxJ.*(Dr*σx)
    rhsσ = dσxdx + LIFT*(σflux)

    # compute df(u)/dx (or u*(du/dx))
    flux = @. u^2/2
    dfdx = rxJ.*(Dr*flux)
    flux_f = Vf*flux
    df = flux_f[mapP] - flux_f
    uflux = @. .5*(df*nxJ - tau*du*abs(.5*(uf[mapP]+uf))*abs(nxJ))
    rhsu = dfdx + LIFT*uflux

    # combine advection and viscous terms
    rhsu = rhsu - ϵ*rhsσ
    return -rhsu./J
end


"Low storage Runge-Kutta time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Perform time-stepping"
u = @. exp(-100*x^2)
# for e = 1:K
#     xavg = sum(x[:,e])/size(x,1)
#     if abs(xavg) < 1/3
#         u[:,e] .= 1
#     else
#         u[:,e] .= 0
#     end
# end

filter_weights = ones(N+1)
# filter_weights[2:end] = exp.(-collect(2:N+1))
filter_weights[end] = 0
Filter = V*(diagm(filter_weights)/V)

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,100))/V
gr(aspect_ratio=1,legend=false,markerstrokewidth=1,markersize=2)

resu = zeros(size(x)) # Storage for the Runge kutta residual storage
interval = 10
@gif for i = 1:Nsteps
    for INTRK = 1:5
        rhsu = rhs(u,ops,vgeo,fgeo,mapP,ϵ,a)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu

        u .= (Filter*u)
    end

    if i%interval==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        plot(Vp*x,Vp*u,ylims=(-.1,1.1),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,u)
    end
end every interval

# scatter!(x,u,markersize=4) # plot nodal values
# plot!(Vp*x,Vp*u) # plot interpolated solution at fine points
