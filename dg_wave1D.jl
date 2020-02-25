using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 3 # The order of approximation
K   = 16
T   = 1 # endtime
CFL = 1
const tau = 1 # upwind penalty parameter

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

"initial conditions"
u0(x) = @. exp(-25*x^2)
p = u0(x)
u = 0*p

p = @. sin(pi*x)
u = 0*p

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Dr,LIFT,Vf)
vgeo = (rxJ,J)
fgeo = (nxJ,)

function rhs(p,u,ops,vgeo,fgeo,mapP)
    # unpack args
    Dr,LIFT,Vf = ops
    rxJ,J = vgeo
    nxJ, = fgeo

    pM = Vf*p # can replace with nodal extraction
    uM = Vf*u # can replace with nodal extraction
    du = uM[mapP]-uM
    dp = pM[mapP]-pM

    px = rxJ*(Dr*p)
    ux = rxJ*(Dr*u)
    rhsp = ux + .5*LIFT*(@. du*nxJ - tau*dp)
    rhsu = px + .5*LIFT*(@. dp*nxJ - tau*du)

    return -rhsp./J, -rhsu./J
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,100))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*u)

resp = zeros(size(x))
resu = zeros(size(x))
@gif for i = 1:Nsteps
    for INTRK = 1:5
        rhsp,rhsu = rhs(p,u,ops,vgeo,fgeo,mapP)
        @. resp = rk4a[INTRK]*resp + dt*rhsp
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. p   += rk4b[INTRK]*resp
        @. u   += rk4b[INTRK]*resu
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        plot(Vp*x,Vp*p,ylims=(-1.1,1.1),title="Timestep $i out of $Nsteps",lw=2)
        scatter!(x,p)
    end
end every 10

rq,wq = gauss_quad(0,0,N+2)
Vq = vandermonde_1D(N,rq)/V
xq = Vq*x
wJq = diagm(wq)*(Vq*J)
err = sqrt(sum(wJq.*(Vq*p - (@. sin(pi*xq)*cos(pi*T))).^2))
@show err
# scatter(x,u,ylims=ulims)
# plot!(Vp*x,u0(Vp*x),ylims=ulims)
