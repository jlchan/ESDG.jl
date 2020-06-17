using Revise # reduce recompilation time
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using ToeplitzMatrices
using Test

using KernelAbstractions
using CUDA
using CUDAapi
CUDA.allowscalar(false)

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

using SetupDG

S_N(x) = @. sin(pi*x/h)/(2*pi/h)/tan(x/2)
"""
Vandermonde matrix of sinc basis functions determined by h,
evaluated at r
"""
function vandermonde_Sinc(h,r)
    N = convert(Int, 2*pi/h)
    V = zeros(length(r),N)
    for n = 1:N
        V[:,n] = S_N(r.-n*h)
    end
    V[1,1] = 1
    V[end,end] = 1
    return V
end

vector_norm(U) = CuArrays.sum((x->x.^2).(U))

"Constants"
const sp_tol = 1e-12
# const has_gpu = CUDAapi.has_cuda_gpu()
const enable_test = false
"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P   = 2;    # The order of approximation in polynomial dimension
Np_P  = Int((N_P+1)*(N_P+2)/2)
Np_F  = 8;    # The order of approximation in Fourier dimension
K1D   = 10;   # Number of elements in polynomial (x,y) dimension
CFL   = 1.0;
T     = 0.5;  # End time

"Time integration Parameters"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N_P+1)*(N_P+2)*3/2  # estimated trace constant for CFL
dt = CFL * 2 / CN / K1D
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Initialize Reference Element in Fourier dimension"
h = 2*pi/Np_F
column = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)]
column2 = [-pi^2/3/h^2-1/6; -((-1).^(1:Np_F-1)./(2*(sin.((1:Np_F-1)*h/2)).^2))]
Dt = Array{Float64,2}(Toeplitz(column,column[[1;Np_F:-1:2]]))
D2t = Array{Float64,2}(Toeplitz(column2,column2[[1;Np_F:-1:2]]))
t = LinRange(h,2*pi,Np_F)


"Initialize Reference Element in polynomial dimension"
rd = init_reference_tri(N_P);
@unpack fv,Nfaces,r,s,VDM,V1,Dr,Ds,rf,sf,wf,nrJ,nsJ,rq,sq,wq,Vq,M,Pq,Vf,LIFT = rd
Nq_P = length(rq)
Nfp_P = length(rf)
Nh_P = Nq_P+Nfp_P # Number of hybridized points
Lq = LIFT

"Mesh related variables"
# First initialize 2D triangular mesh
VX,VY,EToV = uniform_tri_mesh(K1D,K1D)
@. VX = 1+VX
@. VY = 1+VY
md = init_mesh((VX,VY),EToV,rd)
VX = repeat(VX,2)
VY = repeat(VY,2)
VZ = [2/Np_F*ones((K1D+1)*(K1D+1),1); 2*ones((K1D+1)*(K1D+1),1)]
EToV = [EToV EToV.+(K1D+1)*(K1D+1)]

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

# Initialize 3D mesh
@unpack x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapM,mapP,mapB = md
x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = (x->reshape(repeat(x,inner=(1,Np_F)),size(x,1),Np_F,K)).((x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ))
z,zq,zf = (x->reshape(repeat(collect(2/Np_F:(2/Np_F):2),inner=(1,x),outer=(K,1))',x,Np_F,K)).((Np_P,Nq_P,Nfp_P))
mapM = reshape(1:Nfp_P*Np_P*K,Nfp_P,Np_P,K)
mapP_2D = (x->mod1(x,Nfp_P)+div(x-1,Nfp_P)*Nfp_P*Np_F).(mapP)
mapP = reshape(repeat(mapP_2D,inner=(1,Np_F)),Nfp_P,Np_F,K)
for j = 1:Np_F
    mapP[:,j,:] = mapP[:,j,:].+(j-1)*Nfp_P
end

# scale by Fourier dimension
M = h*M
wq = h*wq
wf = h*wf

# Hybridized operators
Vh = [Vq;Vf]
rxJ,sxJ,ryJ,syJ = (x->mapslices((y->Vh*y),x,dims=(1,2))).((rxJ,sxJ,ryJ,syJ))
Ef = Vf*Pq
Br,Bs = (x->diagm(wf.*x)).((nrJ,nsJ))
Qr,Qs = (x->Pq'*M*x*Pq).((Dr,Ds))
Qrh,Qsh = (x->1/2*[x[1]-x[1]' Ef'*x[2];
                   -x[2]*Ef   x[2]]).(((Qr,Br),(Qs,Bs)))
Qrh_skew,Qsh_skew = (x->1/2*(x-x')).((Qrh,Qsh))
Qt = Dt
Qth = Qt # Not the SBP operator, weighted when flux differencing
Ph = [Vq;Vf]*Pq # TODO: refactor

# TODO: assume mesh uniform affine, so Jacobian are constants
# TODO: fix other Jacobian parts
JP = 1/K1D^2
JF = 1/pi
J = JF*JP
wq = J*wq
wf = JF*wf
Lq = 1/JP*Lq
Qrh = JF*Qrh
Qsh = JF*Qsh
Qth = JP*Qth
Qrh_skew = 1/2*(Qrh-Qrh')
Qsh_skew = 1/2*(Qsh-Qsh')
LIFTq = Vq*Lq
VPh = Vq*[Pq Lq]*diagm(1 ./ [wq;wf])

# TODO: refactor
Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,rxJ,sxJ,ryJ,syJ,sJ,nxJ,nyJ,VPh = (x->CuArray{Float32}(x)).((Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,rxJ,sxJ,ryJ,syJ,sJ,nxJ,nyJ,VPh))
ops = (Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq,VPh)
mesh = (rxJ,sxJ,ryJ,syJ,sJ,nxJ,nyJ,JP,JF,J,h,mapM,mapP)
param = (K,Np_P,Nfp_P,Np_F,Nq_P,Nh_P)

xq,yq,zq = (x->reshape(x,Nq_P,Np_F*K)).((xq,yq,zq))
ρ_exact(x,y,z,t) = @. 1+0.2*sin(pi*(x+y+z-3/2*t))
ρ = @. 1+0.2*sin(pi*(xq+yq+zq))
u = ones(size(xq))
v = -1/2*ones(size(xq))
w = ones(size(xq))
p = ones(size(xq))
Q_exact(x,y,z,t) = (ρ_exact(x,y,z,t),u,v,w,p)

Q = primitive_to_conservative(ρ,u,v,w,p)
Q = collect(Q)
Q_vec = zeros(Nq_P,Np_F,5,K)
for k = 1:K
    for d = 1:5
        @. Q_vec[:,:,d,k] = Q[d][:,:,k]
    end
end

Q = Q_vec[:]
Q = CuArray(Q_vec)

VU = CUDA.fill(0.0f0,length(Q))
function v_to_u!(Q,VU)
    index = (blockIdx().x-1)*blockDim().x + threadIdx().x
    stride = blockDim().x*gridDim().x
    for i = index:stride:length(VU)
        @inbounds VU[i] = 1.0
    end
end

num_blocks = ceil(Int,length(VU)/256)
@cuda threads=256 blocks = num_blocks v_to_u!(VU)
@test all(Array(VU) .== 1.0f0)
