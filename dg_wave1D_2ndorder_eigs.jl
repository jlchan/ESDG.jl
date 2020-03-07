using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using SparseArrays

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 6 # The order of approximation
K   = 8

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

"Make maps periodic"
mapP[1] = mapM[end]
mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1

"=========== done with mesh setup here ============ "

"pack arguments into tuples"
ops = (Dr,LIFT,Vf)
vgeo = (rxJ,J)
fgeo = (nxJ,)

function rhs_2ndorder(p,ops,vgeo,fgeo,mapP,params...)
    # unpack arguments
    Dr,LIFT,Vf = ops
    rxJ,J = vgeo
    nxJ, = fgeo

    # construct sigma
    pf = Vf*p
    dp = pf[mapP]-pf
    σxflux = @. .5*dp*nxJ
    dpdx = rxJ.*(Dr*p)
    σx = (dpdx + LIFT*σxflux)./J

    # compute div(σ)
    σxf = Vf*σx
    σxP = σxf[mapP]
    pflux = @. .5*(σxP-σxf)*nxJ
    dσxdx = rxJ.*(Dr*σx)

    tau = params[1]
    rhsp = dσxdx + LIFT*(pflux + tau*dp)

    return rhsp./J
end

tau = 1000
u = zeros(N+1,K)
A = zeros((N+1)*K,(N+1)*K)
for i = 1:(N+1)*K
    u[i] = 1
    r = rhs_2ndorder(u,ops,vgeo,fgeo,mapP,tau)
    A[:,i] = r[:]
    u[i] = 0
end
# Mglobal = kron(diagm(J[1,:]),M)
# A = Mglobal*A
#lam,W = eigen(A,Mglobal)
lam,W = eigen(A)
p = sortperm(abs.(lam)) # sort by magnitude
W = real(W[:,p])
lam = real(lam[p])

gr(legend=false,
    markerstrokewidth=1,markersize=4)

# compute exact eigenvalues
n = size(A,2)
lamex = zeros(n)
lamex[1] = 0
for j = 1:n
    if mod(j,2)==0
        lamex[j] = ((j*pi)/2)^2
    else
        lamex[j] = (((j-1)*pi)/2)^2
    end
end
# scatter(1:n,real(lam))
# scatter!(1:n,-lamex)

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,100))/V
plot()
for i = convert(Int,n/2)
    plot!(Vp*x,Vp*reshape(W[:,i],N+1,K),aspect_ratio=1)
end
display(plot!())
