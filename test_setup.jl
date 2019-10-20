push!(LOAD_PATH, "./src")

# "Packages"
using Revise # reduce need for recompile
using Plots
using Documenter
using LinearAlgebra

# "User defined modules"
using Utils
using Basis1D
using Basis2DQuad
using QuadMeshUtils

"Approximation parameters"
N   = 4; # The order of approximation
K1D = 8

"Mesh related variables"
Nfaces       = 4  # number of faces per element
(VX,VY,EToV) = uniform_quad_mesh(K1D,K1D)
K            = size(EToV,1); # The number of element on the mesh we constructed
Nv           = size(VX,1); # Total number of nodes on the mesh
EToE, EToF   = connect_2D(EToV)

rq,sq,wq = quad_nodes_2D(2*N+1)
V = vandermonde_2D(N,rq,sq)
M = transpose(V)*diagm(wq)*V

scatter(rq,sq,wq,zcolor=wq,camera=(0,90))

# test
include("./src/EntropyStableEuler/logmean.jl")
include("./src/EntropyStableEuler/euler_variables.jl")
include("./src/EntropyStableEuler/euler_fluxes.jl")

rhoL = 2; rhoR = 3
uL = .2;   uR = .1
vL = .1;   vR = .2
EL = 2.0;   ER = 2.1
betaL = betafun(rhoL,uL,vL,EL)
betaR = betafun(rhoR,uR,vR,ER)
UL = (rhoL,uL,vL,betaL)
UR = (rhoR,uR,vR,betaR)
