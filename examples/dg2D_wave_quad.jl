push!(LOAD_PATH, "./src")

# "Packages"
using Revise # reduce need for recompile
using Plots
using Documenter
using LinearAlgebra
using SparseArrays

# "User defined modules"
using Utils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

"Approximation parameters"
N = 4  # The order of approximation
K1D = 16

"Mesh related variables"
(VX, VY, EToV) = uniform_quad_mesh(K1D, K1D)
fv = quad_face_vertices()
Nfaces = length(fv)  # number of faces per element
K  = size(EToV, 1); # The number of element on the mesh we constructed
Nv = size(VX, 1); # Total number of nodes on the mesh
EToE, EToF, FToF = connect_mesh(EToV,fv)

"Set up reference element nodes and operators"
r, s = nodes_2D(N)
V = vandermonde_2D(N, r, s)
Vr, Vs = grad_vandermonde_2D(N, r, s)
Dr = Vr / V
Ds = Vs / V

"Apply mass lumping"
M = inv(V*transpose(V))
M = diagm(vec(sum(M,dims=2)))

"Reference face nodes and normals"
r1D,w1D = gauss_lobatto_quad(0,0,N)
Nfp = length(r1D)
e = ones(size(r1D))
z = zeros(size(r1D))
rf = [r1D; e; -r1D; -e]
sf = [-e; r1D; e; -r1D]
wf = vec(repeat(w1D,Nfaces,1));
nrJ = [z; e; z; -e]
nsJ = [-e; z; e; z]
Vf = vandermonde_2D(N,rf,sf)/V

"Lift matrix"
LIFT = M\(transpose(Vf)*diagm(wf))

# sparsify
Dr = droptol!(sparse(Dr), 1e-10)
Ds = droptol!(sparse(Ds), 1e-10)
LIFT = droptol!(sparse(LIFT),1e-10)

"Map to physical nodes"
r1,s1 = nodes_2D(1)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"Face nodes and connectivity maps"
xf = Vf*x
yf = Vf*y
mapM, mapP, mapB = build_node_maps((xf,yf), FToF)

"Make periodic"
LX = maximum(VX)-minimum(VX)
LY = maximum(VY)-minimum(VY)
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces,mapM,mapP,mapB)
mapP[mapB] = mapPB

"Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ
sJ = @. sqrt(nxJ^2 + nyJ^2)

"initial conditions"
p = @. exp(-100*(x^2+(y-.25)^2))
u = zeros(size(x))
v = zeros(size(x))

"Time integration"
rk4a, rk4b, rk4c = rk45_coeffs()

CN = (N+1)*(N+2)  # estimated trace constant
CFL = .75;
dt = CFL * 2 / (CN*K1D)
T = .75 # endtime
Nsteps = convert(Int,ceil(T/dt))

"pack arguments into tuples"
Q = (p,u,v)
resQ = (zeros(size(x)) for i in eachindex(Q))
ops = (Dr,Ds,LIFT,Vf)
geo = (rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)
nodemaps = (mapP,mapB)

"varying wavespeed"
c2 = @. 1 + .5*sin(pi*x)*sin(pi*y)
# c2 = ones(size(x))

function rhs(Q,ops,geo,nodemaps,params...)

    (p,u,v) = Q
    (Dr,Ds,LIFT,Vf)=ops # should make functions for these
    (rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ)=geo
    (mapP,mapB) = nodemaps
    QM = (x->Vf*x).(Q)
    QP = (x->x[mapP]).(QM)
    (dp,du,dv) = QP.-QM

    "compute central numerical flux"
    (pM,uM,vM) = QM
    # du[mapB] = -2*uM[mapB]
    # dv[mapB] = -2*vM[mapB]
    pflux = @. du*nxJ + dv*nyJ
    uflux = @. dp*nxJ
    vflux = @. dp*nyJ

    pr,ur,vr = (x->Dr*x).(Q)
    ps,us,vs = (x->Ds*x).(Q)

    px = @. rxJ*pr + sxJ*ps;
    py = @. ryJ*pr + syJ*ps
    ux = @. rxJ*ur + sxJ*us;
    vy = @. ryJ*vr + syJ*vs

    rhsp = (ux+vy) + .5*LIFT*pflux
    rhsu = px + .5*LIFT*uflux
    rhsv = py + .5*LIFT*vflux

    c2 = params[1]
    rhsp = @. -c2*rhsp/J
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
