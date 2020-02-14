using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 3 # The order of approximation
K   = 16
T   = 10 # endtime
CFL = 1
const tau = 1 # upwind penalty parameter

"Define mesh and compute connectivity
- (VX,VY) are and EToV is a connectivity matrix
such that EToV[e,:] = vertex indices of element e"
VX = LinRange(-1,1,K+1)
EToV = repeat([0 1],K,1) + repeat(1:K,1,2)

"Construct matrices on reference elements
- r is a vector of interpolation nodes
- V is the matrix arising in polynomial interpolation, e.g. solving V*u = [f(x_1),...,f(x_Np)]
- inv(V) transforms from a nodal basis to an orthonormal basis.
- If Vr evaluates derivatives of the orthonormal basis at nodal points
- then, Vr*inv(V) = Vr/V transforms nodal values to orthonormal coefficients, then differentiates the orthonormal basis"
r,w = gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, r)
Vr = grad_vandermonde_1D(N, r)
Dr = Vr/V
M = inv(V*V') # identity from homework

"Nodes on faces of the reference interval are at -1 and 1"
Vf = vandermonde_1D(N,[-1;1])/V
LIFT = M\(transpose(Vf)) # lift matrix appearing in the RHS evaluation

"Construct global coordinates
- vx = VX[EToV'] = a 2xK matrix of vertex values for each element
- V1*vx uses the linear polynomial defined by 2 vertex values to
interpolate nodal points on the reference interval to each physical interval"
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
x = V1*VX[transpose(EToV)]

"Compute connectivity maps:
- uf = face values of u
- uf[mapP] = uP = exterior value of solution used in DG numerical fluxes"
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

"Make maps periodic
- last node connects back to first node
- first node connects to last node"
mapP[1] = mapM[end]
mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1

"Define initial conditions
- evaluating u0 at nodal points x implicitly defines
a polynomial interpolant of the solution on each element"
u0(x) = @. exp(-25*x^2)
# u0(x) = @. sin(2*pi*x)
# u0(x) = Float64.(@. abs(x) < .33)
u = u0(x)

"Low storage Runge-Kutta time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Dr,LIFT,Vf)
vgeo = (rxJ,J)
fgeo = (nxJ,)

"Define function to evaluate the RHS"
function rhs(u,ops,vgeo,fgeo,mapP)
    # unpack args
    Dr,LIFT,Vf = ops
    rxJ,J = vgeo
    nxJ, = fgeo

    uM = Vf*u # can replace with nodal extraction
    du = uM[mapP]-uM

    ux = rxJ*(Dr*u)
    rhsu = ux + .5*LIFT*(@. du*nxJ - tau*abs(nxJ)*du)

    return -rhsu./J
end

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,100))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*u)

"Perform time-stepping"
resu = zeros(size(x)) # Storage for the Runge kutta residual storage
# @gif
for i = 1:Nsteps
    for INTRK = 1:5
        rhsu = rhs(u,ops,vgeo,fgeo,mapP)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        # plot(Vp*x,Vp*u,ylims=(-.1,1.1),title="Timestep $i out of $Nsteps",lw=2)
        # scatter!(x,u)
    end
end #every 50

scatter(x,u,markersize=4) # plot nodal values
plot!(Vp*x,u0(Vp*x)) # plot interpolated solution at fine points
