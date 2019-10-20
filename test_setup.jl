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
using QuadMeshUtils
using EntropyStableEulerRoutines

"Approximation parameters"
N   = 4; # The order of approximation
K1D = 8

"Mesh related variables"
Nfaces       = 4  # number of faces per element
(VX,VY,EToV) = uniform_quad_mesh(K1D,K1D)
K            = size(EToV,1); # The number of element on the mesh we constructed
Nv           = size(VX,1); # Total number of nodes on the mesh
EToE, EToF   = connect_2D(EToV)

r,s = nodes_2D(N)
V = vandermonde_2D(N,r,s)
rq,sq,wq = quad_nodes_2D(N)
Vq = vandermonde_2D(N,rq,sq)/V
M = transpose(Vq)*diagm(wq)*Vq
Pq = M\(transpose(Vq)*diagm(wq))

r1D,w1D = gauss_quad(0,0,N)
e = ones(size(r1D))
z = zeros(size(r1D))
rf = [r1D; e; -r1D; -e]
sf = [-e; r1D; e; -r1D]
wf = vec(repeat(w1D,Nfaces,1));
nrJ = [z; e; z; -e]
nsJ = [-e; z; e; z]
Vf = vandermonde_2D(N,rf,sf)/V

Vr,Vs = grad_vandermonde_2D(N,r,s)
Dr = Vr/V
Ds = Vs/V
Qr = Pq'*M*Dr*Pq
Qs = Pq'*M*Ds*Pq
E = Vf*Pq
Br = diagm(wf.*nrJ)
Bs = diagm(wf.*nsJ)
Qrh = [Qr-Qr' E'*Br;
-Br*E Br]
Qsh = [Qs-Qs' E'*Br;
-Bs*E Bs]
Qrh = droptol!(sparse(Qrh),1e-10)
Qsh = droptol!(sparse(Qsh),1e-10)

# gr(size=(300,300),legend=false,markerstrokewidth=4,markersize=8)
# gr()
# scatter(rq,sq,camera=(0,90))
# scatter!(rf,sf,camera=(0,90))

# # test
# rhoL = 2.0; rhoR = 3.0
# uL = .2;   uR = .1
# vL = .1;   vR = .2
# EL = 2.0;   ER = 2.1
# betaL = betafun(rhoL,uL,vL,EL)
# betaR = betafun(rhoR,uR,vR,ER)
# UL = (rhoL,uL,vL,betaL)
# UR = (rhoR,uR,vR,betaR)
# U = (rhoL,rhoL*uL,rhoL*vL,EL)

print("Done")
