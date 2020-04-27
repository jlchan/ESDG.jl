using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using SparseArrays
using UnPack

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils
using Basis2DTri
using UniformTriMesh

using Setup2DTri
using UnPack

"Define approximation parameters"
N   = 3 # The order of approximation
K1D = 4 # number of elements along each edge of a rectangle

"=========== Setup code ============="

# construct mesh
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)

# intialize reference operators
rd = init_reference_tri(N)

# initialize physical mesh data
md = init_tri_mesh((VX,VY),EToV,rd)

function rhs(u,rd::RefElemData,md::MeshData,tau)
    # unpack arguments
    @unpack Dr,Ds,LIFT,Vf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP,mapB = md

    # construct sigma
    uf = Vf*u
    uP = uf[mapP]
    uP[mapB] = -uf[mapB] # Dirichlet BCs
    du = uP-uf
    σxflux = @. .5*du*nxJ
    σyflux = @. .5*du*nyJ
    dudxJ = rxJ.*(Dr*u) + sxJ.*(Ds*u)
    dudyJ = ryJ.*(Dr*u) + syJ.*(Ds*u)
    σx = (dudxJ + LIFT*σxflux)./J
    σy = (dudyJ + LIFT*σyflux)./J

    # compute div(σ)
    dσxdx = rxJ.*(Dr*σx) + sxJ.*(Ds*σx)
    dσydy = ryJ.*(Dr*σy) + syJ.*(Ds*σy)
    divσ  = dσxdx + dσydy

    # define local DG flux
    σxf,σyf = (x->Vf*x).((σx,σy))
    σxavg,σyavg = (x->.5*(x[mapP]+x)).((σxf,σyf))
    # TODO: add IPDG flux for hw

    uflux = @. ((σxavg-σxf)*nxJ + (σyavg-σyf)*nyJ) + tau*du
    rhs = divσ + LIFT*uflux

    return rhs./J
end

# build operator
@unpack x = md
tau = 100
Np,K = size(x)
u = zeros(size(x))
A = zeros(Np*K,Np*K)
for i = 1:Np*K
    u[i] = 1
    A[:,i] = rhs(u,rd,md,tau)[:]
    u[i] = 0
end

# multiply by global mass matrix for symmetry
@unpack Vq,wq = rd
@unpack J = md
M = Vq'*diagm(wq)*Vq
A = kron(spdiagm(0=>J[1,:]),M) * A
A = droptol!(sparse(A),1e-12)
nnzA = nnz(A)
spy(A, ms=3, title="nnz = $nnzA")
