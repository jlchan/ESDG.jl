push!(LOAD_PATH, "./src")

# "Packages"
using Revise # reduce recompilation
using Plots
using LinearAlgebra
using SparseArrays

# "User defined modules"
using Utils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

"Approximation parameters"
N = 2 # The order of approximation
K1D = 16

"Mesh related variables"
(VX, VY, EToV) = uniform_quad_mesh(K1D, K1D)
FToF = connect_mesh(EToV,quad_face_vertices())
Nfaces, K = size(FToF)

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

"Map to physical nodes"
r1,s1 = nodes_2D(1)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"Face nodes and connectivity maps"
xf = Vf*x
yf = Vf*y
mapM, mapP, mapB = build_node_maps((xf, yf), FToF)
mapM = reshape(mapM,Nfp*Nfaces,K)
mapP = reshape(mapP,Nfp*Nfaces,K)

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

"initial conditions"
u = @. -sin(pi*x)*sin(pi*y)

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()

"Estimate timestep"
CN = (N+1)*(N+2)  # estimated trace constant
CFL = .75;
dt = CFL * 2 / (CN*K1D)
T = .5 # endtime
Nsteps = convert(Int,ceil(T/dt))

"convert to Gauss node basis"
u = Vq*u
Vh = droptol!(sparse([diagm(ones(length(rq))); E]),1e-10)
Ph = droptol!(sparse(diagm(@. 1/wq)*transpose(Vh)),1e-10)
Lf = droptol!(sparse(diagm(@. 1/wq)*(transpose(E)*diagm(wf))),1e-10)

"Pack arguments into tuples"
ops = (Qrhskew,Qshskew,Ph,Lf)
(rxJh,sxJh,ryJh,syJh) = (x->Vh*x).((rxJ,sxJ,ryJ,syJ))
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

function rhs(Qh,ops,geo,nodemaps)
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
    uflux = @. burgers_flux.(uM,uP)*(nxJ) - .5*lam*du*sJ
    rhsu = Lf*uflux

    # compute volume contributions
    for e = 1:size(u,2)
        Qxh = rxJ[1,e]*Qrhskew + sxJ[1,e]*Qshskew
        rhsu[:,e] += 2*Ph*sparse_hadamard_sum(Qxh,uh[:,e],burgers_flux)
    end

    return -rhsu./J
end

resu = zeros(size(x))
for i = 1:Nsteps
    # global u, resu # for scoping - these variables are updated

    for INTRK = 1:5
        Qh = (Vh*u)
        rhsu = rhs(Qh,ops,geo,nodemaps)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u    = u + rk4b[INTRK]*resu
    end

    if i%10==0 || i==Nsteps
        println("Time step: ", i, " out of ", Nsteps)
    end
end

u = Pq*u

"plotting nodes"
rp, sp = equi_nodes_2D(25)
Vp = vandermonde_2D(N,rp,sp)/V

# pyplot(size=(200,200),legend=false,markerstrokewidth=0,markersize=2)
gr(size=(300,300),legend=false,markerstrokewidth=0,markersize=2)
vv = Vp*u
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
