push!(LOAD_PATH, "./src")

# "Packages"
using Revise # reduce recompilation
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
N = 3 # The order of approximation
K1D = 8

"Mesh related variables"
(VX, VY, EToV) = uniform_quad_mesh(K1D, K1D)
Nfaces = 4  # number of faces per element
K  = size(EToV, 1); # The number of element on the mesh we constructed
Nv = size(VX, 1); # Total number of nodes on the mesh
# EToE, EToF = connect_2D(EToV)
EToE, EToF = connect_mesh(EToV,quad_face_vertices())

"Set up reference element nodes and operators"
r, s = nodes_2D(N)
V = vandermonde_2D(N, r, s)
Vr, Vs = grad_vandermonde_2D(N, r, s)
Dr = Vr / V
Ds = Vs / V

"Quadrature operators"
rq,sq,wq = quad_nodes_2D(N)
Vq = vandermonde_2D(N,rq,sq)/V
M = transpose(Vq)*diagm(wq)*Vq
Pq = M\(transpose(Vq)*diagm(wq))

"Reference face nodes and normals"
# r1D = gauss_lobatto_quad(0,0,N)
# V1D = vandermonde_1D(N,r1D) # hack together face quad nodes
# w1D = vec(sum(inv(V1D*transpose(V1D)),dims=2))
r1D,w1D = gauss_quad(0,0,N)
e = ones(size(r1D))
z = zeros(size(r1D))
rf = [r1D; e; -r1D; -e]
sf = [-e; r1D; e; -r1D]
wf = vec(repeat(w1D,Nfaces,1));
nrJ = [z; e; z; -e]
nsJ = [-e; z; e; z]
Vf = vandermonde_2D(N,rf,sf)/V

"Make hybridized SBP operators"
Qr = Pq'*M*Dr*Pq
Qs = Pq'*M*Ds*Pq
E = Vf*Pq
Br = diagm(wf.*nrJ)
Bs = diagm(wf.*nsJ)
Qrh = .5*[Qr-Qr' E'*Br;
-Br*E Br]
Qsh = .5*[Qs-Qs' E'*Br;
-Bs*E Bs]

"Lift matrix"
Lf = M\(transpose(Vf)*diagm(wf))

"interpolation to and from hybridized quad points"
Vh = [Vq; Vf]
Ph = M\transpose(Vh) # inverse mass for Gauss points

"sparse skew symmetric versions of the operators"
Qrhskew = .5*(Qrh-Qrh')
Qshskew = .5*(Qsh-Qsh')
Qrhskew = droptol!(sparse(Qrhskew),1e-10)
Qshskew = droptol!(sparse(Qshskew),1e-10)
Vh = droptol!(sparse(Vh),1e-10)
Ph = droptol!(sparse(Ph),1e-10)
Lf = droptol!(sparse(Lf),1e-10)

"Map to physical nodes"
r1,s1 = nodes_2D(1)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"Face nodes and connectivity maps"
xf = Vf*x
yf = Vf*y
mapM, mapP, mapB = build_node_maps(xf, yf, Nfaces, EToE, EToF)

"Make node maps periodic"
LX = maximum(VX)-minimum(VX)
LY = maximum(VY)-minimum(VY)
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces,mapM,mapP,mapB)
mapP[mapB] = mapPB

"Geometric factors and surface normals"
rxJ, sxJ, ryJ, syJ, J = geometric_factors(x, y, Dr, Ds)
nxJ = (Vf*rxJ).*nrJ + (Vf*sxJ).*nsJ;
nyJ = (Vf*ryJ).*nrJ + (Vf*syJ).*nsJ;
sJ = @. sqrt(nxJ^2 + nyJ^2)

"Hybridized geofacs"
rxJh = Vh*rxJ; sxJh = Vh*sxJ
ryJh = Vh*ryJ; syJh = Vh*syJ

"initial conditions"
u = @. -sin(pi*x)*sin(pi*y)

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

"Estimate timestep"
CN = (N+1)*(N+2)  # estimated trace constant
CFL = .75;
dt = CFL * 2 / (CN*K1D)
T = .5 # endtime
Nsteps = convert(Int,ceil(T/dt))

rhsu = zeros(size(x))
resu = zeros(size(x))

"Pack arguments into tuples"
ops = (Qrhskew,Qshskew,Ph,Lf)
geo = (rxJh,sxJh,ryJh,syJh,J,nxJ,nyJ,sJ)
Nfp = length(r1D)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)
nodemaps = (mapP,mapB)

function burgers_flux(uL,uR)
    return (uL^2 + uL*uR + uR^2)/6.0
end

"dense lin alg hadamard sum"
function hadamard_sum(A,u,fun)
    ux,uy = meshgrid(u,u)
    return sum(A.*fun.(ux,uy),dims=2)
end

"sparse evaluation of F in sum(Q.*F,dims=2)"
function sparse_hadamard_sum(A,u,fun)
    N = size(A,1)
    AF = zeros(N)
    for i = 1:N
        Ai = A[i,:]
        ui = u[i]
        AFi = 0.0
        for j = Ai.nzind
            uj = u[j]
            AFi += Ai[j]*fun(ui,uj)
        end
        AF[i] = AFi
    end
    return AF
end

function rhs(Qh,ops,geo,nodemaps,hadamard_sum_fun)
    # unpack args
    (uh)=Qh
    (Qrhskew,Qshskew,Ph,Lf)=ops
    (rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ)=geo
    (mapP,mapB) = nodemaps
    Nh = size(Qrhskew,1)
    Nq = size(Ph,1)

    # compute fluxes
    uM = uh[Nq+1:end,:]
    uP = uM[mapP]
    du = uP-uM
    lam = @. max(abs(uM),abs(uP))
    uflux = @. burgers_flux.(uM,uP)*nxJ - .5*lam*du*sJ
    rhsu = Lf*uflux

    # compute volume contributions
    for e = 1:size(u,2)
        Qxh = rxJ[1,e]*Qrhskew + sxJ[1,e]*Qshskew
        rhsu[:,e] += 2*Ph*hadamard_sum_fun(Qxh,uh[:,e],burgers_flux)
    end

    rhsu = @. -rhsu/J
    return (rhsu)
end

# Qh = (Vh*u)
# time = zeros(2)
# for i = 1:10
#     global timea, timeb
#     a = @timed rhs(Qh,ops,geo,nodemaps,hadamard_sum)
#     b = @timed rhs(Qh,ops,geo,nodemaps,sparse_hadamard_sum)
#     time[1] += a[2]
#     time[2] += b[2]
# end
# print("times = ",time/10,"\n")
# error("d")

for i = 1:Nsteps
    global u, resu # for scoping - these variables are updated

    for INTRK = 1:5
        Qh = (Vh*u)
        rhsu = rhs(Qh,ops,geo,nodemaps,sparse_hadamard_sum)
        resu = @. rk4a[INTRK]*resu + dt*rhsu
        u    = @. u + rk4b[INTRK]*resu
    end

    if i%10==0 || i==Nsteps
        println("Time step: ", i, " out of ", Nsteps)
    end
end

"plotting nodes"
rp, sp = equi_nodes_2D(25)
Vp = vandermonde_2D(N,rp,sp)/V

# pyplot(size=(200,200),legend=false,markerstrokewidth=0,markersize=2)
gr(size=(300,300),legend=false,markerstrokewidth=0,markersize=2)

vv = Vp*u
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
