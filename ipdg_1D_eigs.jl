using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using SparseArrays

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 4 # The order of approximation
K   = 8
CFL = 1
T   = 2

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

# "Make maps periodic"
# mapP[1] = mapM[end]
# mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1

"=========== done with mesh setup here ============ "

"pack arguments into tuples"
ops = (Dr,LIFT,Vf)
vgeo = (rxJ,J)
fgeo = (nxJ,)

function rhs(u,ops,vgeo,fgeo,mapP)
    # unpack arguments
    Dr,LIFT,Vf = ops
    rxJ,J = vgeo
    nxJ, = fgeo

    # construct sigma
    uf = Vf*u
    uP = uf[mapP]

    # enforce zero Dirichlet BCs
    uP[1] = -uf[1]
    uP[end] = -uf[end]

    # compute u flux
    uhat = .5*(uP+uf)

    # compute sigma
    σxflux = @. (uhat-uf)*nxJ
    dudxJ = rxJ.*(Dr*u)
    σx = -(dudxJ + LIFT*σxflux)./J

    σxf   = Vf*σx
    uxf   = Vf*(dudxJ./J)

    # compute sigma flux
    tau = 100
    σhat  = .5*(σxf[mapP]+σxf) - tau*(uP-uf).*nxJ
    σhat  = -.5*(uxf[mapP]+uxf) - tau*(uP-uf).*nxJ

    dsig  = σhat-σxf
    σflux = @. (dsig*nxJ)
    dσxdx = rxJ.*(Dr*σx)
    rhs   = dσxdx + LIFT*(σflux)

    return rhs./J
end

u = zeros(N+1,K)
A = zeros((N+1)*K,(N+1)*K)
for i = 1:(N+1)*K
    u[i] = 1
    r = rhs(u,ops,vgeo,fgeo,mapP)
    A[:,i] = r[:]
    u[i] = 0
end
Mglobal = kron(diagm(J[1,:]),M)
A = droptol!(sparse(Mglobal*A),1e-10)
A = Matrix(A)
lam = eigvals(Symmetric(A),Symmetric(Mglobal))

@show norm(A-A')
@show minimum(lam)

j = vec(1:size(A,2))
scatter(j,j.^2*pi^2/4) # min eigvalue
scatter!(j,sort(real(lam)))
