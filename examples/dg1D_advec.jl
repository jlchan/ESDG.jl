using Revise # reduce need for recompile
using Plots
using Documenter
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using Utils
using Basis1D

"Approximation parameters"
N   = 3 # The order of approximation
K1D = 32
CFL = .75
T   = 10 # endtime

"Mesh related variables"
VX = LinRange(-1,1,K1D+1)
EToV = transpose(reshape(sort([1:K1D; 2:K1D+1]),2,K1D))

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
mapM = reshape(1:2*K1D,2,K1D)
mapP = copy(mapM)
for e = 1:K1D
    if e > 1
        mapP[1,e] = mapM[2,e-1]
    end
    if e < K1D
        mapP[2,e] = mapM[1,e+1]
    end
end

"Make periodic"
mapP[1] = mapM[end]
mapP[end] = mapM[1]

"Geometric factors and surface normals"
h = diff(VX)
J = repeat(transpose(h/2),N+1,1)
nxJ = repeat([-1;1],1,K1D)
rxJ = 1

"initial conditions"
u = @. exp(-25*x^2)

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()

CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Dr,LIFT,Vf)
vgeo = (rxJ,J)
fgeo = (nxJ)

function rhs(u,ops,vgeo,fgeo,mapP)

    # unpack args
    Dr,LIFT,Vf = ops
    rxJ,J = vgeo
    nxJ = fgeo

    uM = Vf*u # can replace with nodal extraction
    du = uM[mapP]-uM

    ux = rxJ*(Dr*u)
    tau = 1 # upwind penalty parameter
    rhsu = ux + .5*LIFT*(@. du*nxJ - tau*abs(nxJ)*du)

    return -rhsu./J
end

resu = zeros(size(x))

for i = 1:Nsteps
    global u, resu # for scoping - these variables are updated

    for INTRK = 1:5
        rhsu = rhs(u,ops,vgeo,fgeo,mapP)
        resu = @. rk4a[INTRK]*resu + dt*rhsu
        u    = @. u + rk4b[INTRK]*resu
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

"plotting nodes"
rp = LinRange(-1,1,10)
Vp = vandermonde_1D(N,rp)/V

# pyplot(size=(500,500),legend=false,markerstrokewidth=0)
gr(size=(300,300),legend=false,markerstrokewidth=0,markersize=2)
plot(Vp*x,Vp*u)
