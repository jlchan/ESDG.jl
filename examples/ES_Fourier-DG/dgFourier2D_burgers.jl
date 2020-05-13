using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack
using ToeplitzMatrices

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

using SetupDG


# TODO: refactor the code
function fS(uL, uR)
    return 1/6*(uL^2+uR^2+uL*uR)
end

function burgers_exact_sol_2D(u0,x,y,T,dt)
    Nsteps = ceil(Int,T/dt)
    dt = T/Nsteps
    u = u0(x,y) # computed at input points
    for i = 1:Nsteps
        t = i*dt
        u .= @. u0(x-u*t,y) # evolve solution at quadrature points using characteristics
    end
    return u
end

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
    return V
end


"Constants"
const sp_tol = 1e-12
u0(x,y) = @. -sin(pi*x)*sin(pi*y)
# u0(x,y) = @. exp(-1*(y^2+x^2))


"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P = 2;    # The order of approximation in polynomial dimension
Np_P = N_P+1;
Np_F = 8;    # The order of approximation in Fourier dimension
K    = 30;   # Number of elements in polynomial dimension
CFL  = 0.1;
T    = 1.0;  # End time

"Time integration Parameters"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N_P+1)*(N_P+2)/2  # estimated trace constant for CFL
dt = CFL * 2 / CN / K
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Initialize Reference Element in Fourier dimension"
h = 2*pi/Np_F
column = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)];
Ds = Toeplitz(column,column[[1;Np_F:-1:2]]);
s = LinRange(h,2*pi,Np_F)

"Mesh related variables"
VX = repeat(LinRange(-1,1,K+1),2)
VY = [(-1+2/Np_F)*ones(K+1,1);ones(K+1,1)]
fv = quad_face_vertices()
Nfaces = length(fv)
EToV = hcat(1:K, K+2:2*K+1, 2:K+1, K+3:2*K+2)

"Initialize Reference Element in polynomial dimension"
rd = init_reference_interval(N_P);
@unpack r,rq,wq,rf,Dr,VDM,Vq,Vf,Vp,M,Pq,LIFT,nrJ = rd
wf = [1;1]
Nq_P = length(rq)
Nfp_P = length(rf)
Nh_P = Nq_P+Nfp_P # Number of hybridized points
rp = LinRange(-1,1,100)
Vp = vandermonde_1D(N_P,rp)/VDM # Redefine Vp
Lq = LIFT

"Initialize Mesh"
FToF = connect_mesh(EToV,fv)
r1 = [-1; -1; 1; 1]
s1 = [h; 2*pi; h; 2*pi]
@unpack r = rd
s,r = meshgrid(LinRange(h,2*pi,Np_F),r)
V1 = vandermonde_2D(1,r,s)/vandermonde_2D(1,r1,s1)
x = reshape(V1*VX[transpose(EToV)],Np_P,Np_F,K)
y = reshape(V1*VY[transpose(EToV)],Np_P,Np_F,K)
mapM = reshape(collect(1:2*K*Np_F),2,K*Np_F)
mapP = [[mapM[2,end-Np_F+1:end]; mapM[2,1:end-Np_F]]'; [mapM[1,Np_F+1:end]; mapM[1,1:Np_F]]']

# Geometric Factors
xr = mapslices((x->Dr*x),x,dims=(1,2))
rxJ = ones(size(xr))
nxJ = repeat(nrJ,1,Np_F,K)
ys = 1/pi*ones(size(xr)) # TODO: hard coded
JP = xr # TODO: redundant, for clarity
JF = ys
J = @. JP*JF
#TODO: only assume uniform affine mesh, need to test for nonuniform meshes
#TODO: a lot of redundancy
JPq,JFq,Jq = (x->mapslices((x->Vq*x),x,dims=(1,2))).((JP,JF,J))
JPf,JFf,Jf = (x->mapslices((x->Vf*x),x,dims=(1,2))).((JP,JF,J))

# scale by Fourier dimension
M = h*M
wq = h*wq
wf = h*wf

# Hybridized operators
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Qr = Pq'*M*Dr*Pq
Qs = Ds
Drh = [Vq*Dr*Pq-1/2*Vq*Lq*diagm(nrJ)*Vf*Pq  1/2*Vq*Lq*diagm(nrJ);
       -1/2*diagm(nrJ)*Vf*Pq                1/2*diagm(nrJ)]
Qrh = 1/2*[Qr-Qr' Ef'*Br;
           -Br*Ef Br]
Qrh_skew = 1/2*(Qrh-Qrh')
Qsh = Qs # Not the SBP operator, weighted when flux differencing


# TODO: assume mesh uniform affine, so Jacobian are constants
# TODO: fix other Jacobian parts
JP = 1/K
JF = 1/pi
J = JF*JP
wq = J*wq
wf = JF*wf
Lq = 1/JP*Lq
Qrh = JF*Qrh
Qsh = JP*Qsh


# TODO: refactor
ops = (Vq,Vf,wq,wf,Pq,Lq,Qrh,Qsh)
mesh = (nrJ,JP,JF,J,h,mapM,mapP)
param = (K,Np_P,Np_F,Nh_P)
function rhs(u,ops,mesh,param,compute_rhstest)
    Vq,Vf,wq,wf,Pq,Lq,Qrh,Qsh = ops
    nrJ,JP,JF,J,h,mapM,mapP = mesh
    K,Np_P,Np_F = param
    uq,uf = (A->A*u).((Vq,Vf))
    uh = [uq;uf]
    w = [wq;wf]
    ∇fh = zeros(size(uh))

    # TODO: fix Jacobian
    # Flux differencing in x direction
    for k = 1:K
        for nf = 1:Np_F
            j = nf+Np_F*(k-1)
            ∇fh[:,j] += 2*Qrh.*[fS(uL,uR) for uL in uh[:,j], uR in uh[:,j]]*ones(size(uh,1),1)
        end
    end

    # Flux differencing in y direction
    for k = 1:K
        for nh = 1:Nh_P
            ∇fh[nh,(k-1)*Np_F+1:k*Np_F] += 2*[wq/J;wf/JF][nh]*Qsh.*[fS(uL,uR) for uL in uh[nh,(k-1)*Np_F+1:k*Np_F], uR in uh[nh,(k-1)*Np_F+1:k*Np_F]]*ones(Np_F,1)
        end
    end

    # Spatial term
    ∇f = [Pq Lq]*diagm(1 ./ w)*∇fh
    # Flux term
    uf = Vf*u
    uM = uf[mapM]
    uP = uM[mapP]
    LF = @. max(abs(uP),abs(uM))*(uP-uM)
    uflux = diagm(nrJ)*(@. fS(uP,uM)-uM*uM/2)-LF
    rhsu = -(∇f+Lq*uflux)

    rhstest = 0
    if compute_rhstest
        rhstest += sum(diagm(wq)*uq.*(Vq*rhsu))
    end

    return rhsu,rhstest
end


# Time stepping
u = u0(x,y)
u = reshape(u,size(u,1),size(u,2)*size(u,3)) # reshape for faster matrix multiplication
resu = zeros(size(u))
for i = 1:Nsteps
    rhstest = 0
    for INTRK = 1:5
        compute_rhstest = INTRK==5
        rhsu,rhstest = rhs(u,ops,mesh,param,compute_rhstest)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u += rk4b[INTRK]*resu
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")
    end
end


# Plotting
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=3)
plot()

u = reshape(u,Np_P,Np_F,K)
yp = LinRange(-1+2/Np_F,1,200)
sp = LinRange(h,2*pi,200)
VDM_F = vandermonde_Sinc(h,sp)
V2 = vandermonde_1D(1,sp)/vandermonde_1D(1,LinRange(h,2*pi,Np_F))
Vp = vandermonde_1D(N_P,LinRange(-1,1,10))/VDM
x_plot = mapslices((x->Vp*x*V2'),x,dims=(1,2))
y_plot = mapslices((x->Vp*x*V2'),y,dims=(1,2))
u_plot = mapslices((x->Vp*x*VDM_F'),u,dims=(1,2))
plt = scatter(vec(x_plot),vec(y_plot),vec(u_plot),zcolor=vec(u_plot),camera=(0,90))
display(plt)
