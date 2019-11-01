push!(LOAD_PATH, "./src")

# "Packages"
using Revise # reduce need for recompile
using Plots
using Documenter
using LinearAlgebra

# "User defined modules"
using Utils
using Basis1D
using Basis2DTri
using UniformTriMesh

"Approximation parameters"
N   = 3; # The order of approximation
K1D = 16

"Mesh related variables"
(VX,VY,EToV)   = uniform_tri_mesh(K1D,K1D)
fv             = tri_face_vertices()
Nfaces         = length(fv)  # number of faces per element
K              = size(EToV,1); # number of element on the mesh we constructed
Nv             = size(VX,1);   # number of vertex nodes
EToE,EToF,FToF = connect_mesh(EToV,fv)

"Construct matrices on reference elements"
r, s = nodes_2D(N)
V = vandermonde_2D(N, r, s)
M = inv(V*V')
Vr, Vs = grad_vandermonde_2D(N, r, s)
Dr = Vr/V
Ds = Vs/V
Np  = length(r);

"Nodes on faces, and face node coordinate"
r1D, w1D = gauss_quad(0,0,N)
Nfp = length(r1D)
e = ones(Nfp,1)
z = zeros(Nfp,1)
rf = [r1D; -r1D; -e];
sf = [-e; r1D; -r1D];
wf = vec(repeat(w1D,3,1));
nrJ = [z; e; -e]
nsJ = [-e; e; z]
Vf = vandermonde_2D(N,rf,sf)/V

"Lift matrix"
LIFT = M\(transpose(Vf)*diagm(wf))

"Construct global coordinates"
r1,s1 = nodes_2D(1)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"Connectivity maps"
xf = Vf*x
yf = Vf*y
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)

"Make periodic"
LX = maximum(VX)-minimum(VX)
LY = maximum(VY)-minimum(VY)
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces,mapM,mapP,mapB)
mapP[mapB] = mapPB

"Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)

"initial conditions"
p = @. exp(-25*((x-.5)^2+y^2))
u = zeros(size(x))
v = zeros(size(x))

"Time integration"
rk4a = [            0.0 ...
-567301805773.0/1357537059087.0 ...
-2404267990393.0/2016746695238.0 ...
-3550918686646.0/2091501179385.0  ...
-1275806237668.0/842570457699.0];
rk4b = [ 1432997174477.0/9575080441755.0 ...
5161836677717.0/13612068292357.0 ...
1720146321549.0/2090206949498.0  ...
3134564353537.0/4481467310338.0  ...
2277821191437.0/14882151754819.0]
rk4c = [ 0.0  ...
1432997174477.0/9575080441755.0 ...
2526269341429.0/6820363962896.0 ...
2006345519317.0/3224310063776.0 ...
2802321613138.0/2924317926251.0 ...
1.0];

CN = (N+1)*(N+2)/2  # estimated trace constant
CFL = .5;
dt = CFL * 2 / (CN*K1D)
T = 0.75 # endtime
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

rhsp = zeros(size(x))
rhsu = zeros(size(x))
rhsv = zeros(size(x))
resp = zeros(size(x))
resu = zeros(size(x))
resv = zeros(size(x))

"pack arguments into tuples"
Q = (p,u,v)
rhsQ = (rhsp,rhsu,rhsv)
resQ = (resp,resu,resv)
ops = (Dr,Ds,LIFT,Vf)
geo = (rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)
nodemaps = (mapP,mapB)

"varying wavespeed"
c2 = @. 1 + .5*sin(pi*x)*sin(pi*y)
c2 = V*(diagm([1;zeros(Np-1)])*(V\c2)) # approx with a constant
# c2 = ones(size(x))

function rhs(Q,ops,geo,nodemaps,params...)

    (p,u,v) = Q
    (Dr,Ds,LIFT,Vf)=ops # should make functions for these
    (rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ)=geo
    (mapP,mapB) = nodemaps
    pM = Vf*p
    uM = Vf*u
    vM = Vf*v
    pP = pM[mapP]
    uP = uM[mapP]
    vP = vM[mapP]

    dp = pP-pM
    du = uP-uM
    dv = vP-vM

    # "uncomment for wall boundary conditions"
    # dp[mapB] = 0*pM[mapB]
    # du[mapB] = -2*uM[mapB]
    # dv[mapB] = -2*vM[mapB]

    pflux = @. du*nxJ + dv*nyJ
    uflux = @. dp*nxJ
    vflux = @. dp*nyJ

    pr = Dr*p;   ps = Ds*p
    ur = Dr*u;   us = Ds*u
    vr = Dr*v;   vs = Ds*v

    px = @. rxJ*pr + sxJ*ps;
    py = @. ryJ*pr + syJ*ps
    ux = @. rxJ*ur + sxJ*us;
    vy = @. ryJ*vr + syJ*vs

    rhsp = (ux+vy) + .5*LIFT*pflux
    rhsu = px + .5*LIFT*uflux
    rhsv = py + .5*LIFT*vflux

    c2 = params[1]
    rhsp = @. c2*rhsp

    rhsp = @. -rhsp/J
    rhsu = @. -rhsu/J
    rhsv = @. -rhsv/J
    return (rhsp,rhsu,rhsv)
end

for i = 1:Nsteps
    global Q, resQ # for scoping - these variables are updated

    for INTRK = 1:5
        rhsQ = rhs(Q,ops,geo,nodemaps,c2)
        resQ = @. rk4a[INTRK]*resQ + dt*rhsQ
        Q    = @. Q + rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps: ", i, " out of ", Nsteps)
    end
end

(p,u,v) = Q

"plotting nodes"
rp, sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/V

# pyplot(size=(500,500),legend=false,markerstrokewidth=0)
gr(size=(300,300),legend=false,markerstrokewidth=0,markersize=2)

vv = Vp*p
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
