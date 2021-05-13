using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

"Approximation parameters"
N   = 3 # The order of approximation
K1D = 16
CFL = .5
T   = 2.0 # endtime

"Mesh related variables"
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
FToF = connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

"Construct matrices on reference elements"
r, s = nodes_2D(N)
V = vandermonde_2D(N, r, s)
Vr, Vs = grad_vandermonde_2D(N, r, s)
M = inv(V*V')
Dr = Vr/V
Ds = Vs/V

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
LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix

"Construct global coordinates"
r1,s1 = nodes_2D(1)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"Connectivity maps"
xf,yf = (x->Vf*x).((x,y))
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)

"Make periodic"
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

"Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)

"initial conditions"
u = @. exp(-25*(x^2+y^2))

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Dr,Ds,LIFT,Vf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ)
nodemaps = (mapP,mapB)

function rhs(u,ops,vgeo,fgeo,nodemaps)
    # unpack args
    Dr,Ds,LIFT,Vf = ops
    rxJ,sxJ,ryJ,syJ,J = vgeo
    nxJ,nyJ,sJ = fgeo
    (mapP,mapB) = nodemaps

    uM = Vf*u # can replace with nodal extraction
    uP = uM[mapP]
    du = uP-uM

    ur = Dr*u
    us = Ds*u
    ux = @. rxJ*ur + sxJ*us

    tau = .5
    rhsu = ux + LIFT*(@. .5*du*nxJ - tau*du*abs(nxJ))

    return -rhsu./J
end

resu = zeros(size(x))
for i = 1:Nsteps
    global u, resu # for scoping - these variables are updated

    for INTRK = 1:5
        rhsu = rhs(u,ops,vgeo,fgeo,nodemaps)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u += rk4b[INTRK]*resu
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

"plotting nodes"

gr(aspect_ratio=1,legend=false,
   markerstrokewidth=0,markersize=2)

rp, sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/V
vv = Vp*u
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
