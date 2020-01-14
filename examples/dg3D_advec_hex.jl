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

N = 3
K1D = 8
VX,VY,VZ,EToV = uniform_hex_mesh(K1D,K1D,K1D)
FToF = connect_mesh(EToV,hex_face_vertices())
Nfaces, K = size(FToF)

r,s,t = nodes_3D(N)
V = vandermonde_3D(N,r,s,t)
Vr,Vs,Vt = grad_vandermonde_3D(N,r,s,t)
Dr = Vr/V
Ds = Vs/V
Dt = Vt/V

"quadrature"
rq,sq,tq,wq = quad_nodes_3D(N)
Vq = vandermonde_3D(N,rq,sq,tq)/V
M = transpose(Vq)*diagm(wq)*Vq
Pq = M\(transpose(Vq)*diagm(wq))

"face nodes and matrices"
rquad,squad,wquad = Basis2DQuad.quad_nodes_2D(N)
Nfp = length(rquad)
e = ones(size(rquad))
zz = zeros(size(rquad))
rf = [-e; e; rquad; rquad; rquad; rquad]
sf = [rquad; rquad; -e; e; squad; squad]
tf = [squad; squad; squad; squad; -e; e]
wf = vec(repeat(wquad,Nfaces,1));
nrJ = [-e; e; zz;zz; zz;zz]
nsJ = [zz;zz; -e; e; zz;zz]
ntJ = [zz;zz; zz;zz; -e; e]

"surface operators"
Vf = vandermonde_3D(N,rf,sf,tf)/V
Lf = M\(transpose(Vf)*diagm(wf))

"map nodes"
r1,s1,t1 = nodes_3D(1)
V1 = vandermonde_3D(1,r,s,t)/vandermonde_3D(1,r1,s1,t1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]
z = V1*VZ[transpose(EToV)]

"get physical face nodes"
xf = Vf*x
yf = Vf*y
zf = Vf*z
mapM, mapP, mapB = build_node_maps((xf,yf,zf),FToF)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)

"make periodic"
LX = 2; LY = 2; LZ = 2
mapPB = build_periodic_boundary_maps(xf,yf,zf,LX,LY,LZ,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB

"Geometry"
geo = geometric_factors(x,y,z,Dr,Ds,Dt)
(rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ, J) = geo
nxJ = nrJ.*(Vf*rxJ) + nsJ.*(Vf*sxJ) + ntJ.*(Vf*txJ)
nyJ = nrJ.*(Vf*ryJ) + nsJ.*(Vf*syJ) + ntJ.*(Vf*tyJ)
nzJ = nrJ.*(Vf*rzJ) + nsJ.*(Vf*szJ) + ntJ.*(Vf*tzJ)
sJ = @. sqrt(nxJ.^2 + nyJ.^2 + nzJ.^2)

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

CN = (N+1)*(N+2)*3/2  # estimated trace constant
CFL = .75;
dt = CFL * 2 / (CN*K1D)
T = .5 # endtime
Nsteps = convert(Int,ceil(T/dt))


"sparsify"
Dr = droptol!(sparse(Dr),1e-10)
Ds = droptol!(sparse(Ds),1e-10)
Dt = droptol!(sparse(Dt),1e-10)
Vf = droptol!(sparse(Vf),1e-10)
Lf = droptol!(sparse(Lf),1e-10)

"pack arguments into tuples"
ops = (Dr,Ds,Dt,Lf,Vf)
geo = (rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J,nxJ,nyJ,nzJ,sJ)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)
nodemaps = (mapP,mapB)

function rhs(Q,ops,geo,nodemaps,params...)
    u = Q
    (Dr,Ds,Dt,Lf,Vf)=ops # should make functions for these
    (rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J,nxJ,nyJ,nzJ,sJ)=geo
    (mapP,mapB) = nodemaps
    uM = Vf*u
    uP = uM[mapP]

    du = uP-uM
    uflux = @. .5*(du*nxJ - abs(nxJ)*du)

    ur = Dr*u
    us = Ds*u
    ut = Dt*u
    ux = @. rxJ*ur + sxJ*us + txJ*ut

    rhsu = ux + Lf*uflux

    rhsu = @. -rhsu/J
    return (rhsu)
end

"initial conditions"
u = @. sin(pi*x)
u = @. exp(-25*(x^2+y^2))
Q = u
resQ = zeros(size(x))

for i = 1:Nsteps
    # global Q, resQ # for scoping - these variables are updated. TODO: remove

    for INTRK = 1:5
        rhsQ = rhs(Q,ops,geo,nodemaps)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q    = Q + rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps: ", i, " out of ", Nsteps)
    end
end

(u) = Q

"plotting nodes"
rp, sp, tp = equi_nodes_3D(15)
Vp = vandermonde_3D(N,rp,sp,tp)/V

# pyplot(size=(100,100),legend=false,markerstrokewidth=0,markersize=2)
gr(size=(200,200),legend=false,markerstrokewidth=0,markersize=2)

xp = Vp*x
yp = Vp*y
zp = Vp*z
vv = Vp*u

ids = map(x->x[1],findall(@. abs(zp[:])<1e-10))
(xp,yp,zp,vv) = (x->x[ids]).((xp,yp,zp,vv))
scatter(xp,yp,vv,zcolor=vv,camera=(0,90))
