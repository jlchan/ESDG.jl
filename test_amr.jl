push!(LOAD_PATH, "./src")
push!(LOAD_PATH, "./AdaptiveRefinement")
using Revise # reduce recompilation time
using RefinementTree
using Plots
using LinearAlgebra
using SparseArrays
using BenchmarkTools

# "User defined modules"
using Utils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

K1D = 2
(VX, VY, EToV) = uniform_quad_mesh(K1D, 1)
VX = 2*VX
Nfaces = 4  # number of faces per element
K  = size(EToV, 1); # The number of element on the mesh we constructed
Nv = size(VX, 1); # Total number of nodes on the mesh
EToE, EToF, FToF = connect_mesh(EToV,quad_face_vertices())

error("d")

"Set up reference element nodes and operators"
N = 4;
r, s = nodes_2D(N)
V = vandermonde_2D(N, r, s)
r1D,w1D = gauss_quad(0,0,N)
e = ones(size(r1D))
rf = [r1D; e; -r1D; -e]
sf = [-e; r1D; e; -r1D]
wf = vec(repeat(w1D,Nfaces,1));
Vf = vandermonde_2D(N,rf,sf)/V

rq,sq,wq = quad_nodes_2D(N)
Vq = vandermonde_2D(N,rq,sq)/V

"refinement VDM"
rref = [@. -1+.5*(1+r); @.  .5*(1+r);   @. .5*(1+r);  @. -1+.5*(1+r)]
sref = [@. -1+.5*(1+s); @.  -1+.5*(1+s);@.  .5*(1+s); @. .5*(1+s)]
Vref = vandermonde_2D(N,rref,sref)/V

"Map physical nodes"
r1,s1 = nodes_2D(1)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = V1*VX[transpose(EToV)]
y = V1*VY[transpose(EToV)]

"============ refinement tree ==========================="

meshtree = MeshTree(EToE,EToF)
refine!(1,meshtree)
x = hcat(x,reshape(Vref*x[:,1],length(r),4))
y = hcat(y,reshape(Vref*y[:,1],length(r),4))

"Face nodes and connectivity maps"
xf = Vf*x
yf = Vf*y

gr(size=(300,300),legend=false,aspect_ratio=:equal)

active_elems = findall(meshtree.active)
scatter(Vq*x[:,active_elems],Vq*y[:,active_elems])
# scatter(rref,sref)
