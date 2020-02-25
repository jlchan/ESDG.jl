using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

"Define approximation parameters"
N   = 2 # The order of approximation
K1D = 8 # number of elements along each edge of a rectangle
CFL = 1 # relative size of a time-step
T   = 1 # final time

"Define mesh and compute connectivity
- (VX,VY) are and EToV is a connectivity matrix
- connect_mesh computes a vector FToF such that face i is connected to face j=FToF[i]"
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
FToF = connect_mesh(EToV,tri_face_vertices())
Nfaces,K = size(FToF)

# iids = @. (abs(abs(VX)-1)>1e-10) & (abs(abs(VY)-1)>1e-10)
# a = .2/K1D
# VX[iids] = @. VX[iids] + a*randn()
# VY[iids] = @. VY[iids] + a*randn()

"Construct matrices on reference elements
- r,s are vectors of interpolation nodes
- V is the matrix arising in polynomial interpolation, e.g. solving V*u = [f(x_1),...,f(x_Np)]
- inv(V) transforms from a nodal basis to an orthonormal basis.
- If Vr evaluates derivatives of the orthonormal basis at nodal points
- then, Vr*inv(V) = Vr/V transforms nodal values to orthonormal coefficients, then differentiates the orthonormal basis"
r, s = nodes_2D(N)
V = vandermonde_2D(N, r, s)
Vr, Vs = grad_vandermonde_2D(N, r, s)
invM = (V*V')
Dr = Vr/V
Ds = Vs/V

"Nodes on faces, and face node coordinate
- r1D,w1D are quadrature nodes and weights
- rf,sf = mapping 1D quad nodes to the faces of a triangle"
r1D, w1D = gauss_quad(0,0,N)
Nfp = length(r1D) # number of points per face
e = ones(Nfp,1) # vector of all ones
z = zeros(Nfp,1) # vector of all zeros
rf = [r1D; -r1D; -e];
sf = [-e; r1D; -r1D];
wf = vec(repeat(w1D,3,1));
Vf = vandermonde_2D(N,rf,sf)/V # interpolates from nodes to face nodes
LIFT = invM*(transpose(Vf)*diagm(wf)) # lift matrix used in rhs evaluation

"Construct global coordinates
- vx = VX[EToV'] = a 3xK matrix of vertex values for each element
- V1*vx uses the linear polynomial defined by 3 vertex values to
interpolate nodal points on the reference element to physical elements"
r1,s1 = nodes_2D(1)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"Compute connectivity maps: uP = exterior value used in DG numerical fluxes"
xf,yf = (x->Vf*x).((x,y))
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)

"Make boundary maps periodic"
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

"Compute geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)

"nhat = (nrJ,nsJ) are reference normals scaled by edge length
- physical normals are computed via G*nhat, G = matrix of geometric terms
- sJ is the normalization factor for (nx,ny) to be unit vectors"
nrJ = [z; e; -e]
nsJ = [-e; e; z]
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)

"=========== Done defining geometry and mesh ============="

"Define the initial conditions by interpolation"
#pex = (x,y,t)->@. cos(pi/2*x)*cos(pi/2*y)*cos(sqrt(2)/2*pi*t)
k = 1
pex = (x,y,t)->@. sin(k*pi*x)*sin(k*pi*y)*cos(sqrt(2)*k*pi*t)

p = pex(x,y,0)
# p = @. exp(-250*(x^2+y^2))
u = @. 0*x
v = @. 0*x

"Time integration coefficients"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"pack arguments into tuples"
ops = (Dr,Ds,LIFT,Vf)
vgeo = (rxJ,sxJ,ryJ,syJ,J)
fgeo = (nxJ,nyJ,sJ)
mapP = reshape(mapP,Nfp*Nfaces,K)
nodemaps = (mapP,mapB)

"Define function to evaluate the RHS: Q = (p,u,v)"
function rhs(Q,ops,vgeo,fgeo,nodemaps)
    # unpack arguments
    Dr,Ds,LIFT,Vf = ops
    rxJ,sxJ,ryJ,syJ,J = vgeo
    nxJ,nyJ,sJ = fgeo
    (mapP,mapB) = nodemaps

    Qf = (x->Vf*x).(Q)
    QP = (x->x[mapP]).(Qf)
    dp,du,dv = QP.-Qf
    tau = .5
    dun = @. (du*nxJ + dv*nyJ)/sJ
    pflux = @. .5*(du*nxJ + dv*nyJ) - tau*dp*sJ
    uflux = @. .5*dp*nxJ - tau*dun*nxJ
    vflux = @. .5*dp*nyJ - tau*dun*nyJ
    # uflux = @. .5*dp*nxJ - tau*du*sJ
    # vflux = @. .5*dp*nyJ - tau*dv*sJ

    pr,ur,vr = (x->Dr*x).(Q)
    ps,us,vs = (x->Ds*x).(Q)
    dudx = @. rxJ*ur + sxJ*us
    dvdy = @. ryJ*vr + syJ*vs
    dpdx = @. rxJ*pr + sxJ*ps
    dpdy = @. ryJ*pr + syJ*ps
    rhsp = dudx + dvdy + LIFT*pflux
    rhsu = dpdx        + LIFT*uflux
    rhsv = dpdy        + LIFT*vflux

    return (x->-x./J).((rhsp,rhsu,rhsv))
end

"Perform time-stepping"
Q = [p,u,v]
resQ = [zeros(size(x)) for i in eachindex(Q)]
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,ops,vgeo,fgeo,nodemaps)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q    = Q + rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    #     p = Q[1]
    #     display(scatter(x,y,p,zcolor=p,zlims=(-1,1)))
    end
end

p,u,v = Q

rq,sq,wq = quad_nodes_2D(2*N+2)
Vq = vandermonde_2D(N,rq,sq)/V
xq,yq = (x->Vq*x).((x,y))
wJq = diagm(wq)*(Vq*J)

@show sum(wJq.*(Vq*p - pex(xq,yq,T)).^2)

"plotting nodes"
gr(size=(300,300),legend=false,markerstrokewidth=0,markersize=2)
rp, sp = equi_nodes_2D(10)
Vp = vandermonde_2D(N,rp,sp)/V
vv = Vp*p
display(scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90),zlims=(-1,1)))
