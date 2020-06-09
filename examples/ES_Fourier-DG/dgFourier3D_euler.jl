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

# For testing TODO: redundant
γ = 1.4
function Unorm(U)
    unorm = zeros(size(U[1]))
    for u in U
        @. unorm += u^2
    end
    return unorm
end
"pressure as a function of ρ,u,v,E"
function pfun(rho,rhoU,E,rhounorm)
    return @. (γ-1)*(E-.5*rhounorm)
end
function pfun(rho,rhoU,E)
    rhounorm = Unorm(rhoU)./rho
    return pfun(rho,rhoU,E,rhounorm)
end

"Constants"
const sp_tol = 1e-12


"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P   = 2;    # The order of approximation in polynomial dimension
Np_P  = Int((N_P+1)*(N_P+2)/2)
Np_F  = 8;    # The order of approximation in Fourier dimension
K1D   = 10;   # Number of elements in polynomial (x,y) dimension
CFL   = 1.0;
T     = 5.0;  # End time

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
Dt = Toeplitz(column,column[[1;Np_F:-1:2]])
D2t = Toeplitz(column2,column2[[1;Np_F:-1:2]])
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
x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ = (x->reshape(repeat(x,inner=(1,Np_F)),size(x,1),Np_F*K)).((x,y,xf,yf,xq,yq,rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ))
z,zq,zf = (x->reshape(repeat(collect(2/Np_F:(2/Np_F):2),inner=(1,x),outer=(K,1))',x,Np_F*K)).((Np_P,Nq_P,Nfp_P))
mapM = reshape(1:Nfp_P*Np_P*K,Nfp_P,Np_P*K)
mapP_2D = (x->mod1(x,Nfp_P)+div(x-1,Nfp_P)*Nfp_P*Np_F).(mapP)
mapP = reshape(repeat(mapP_2D,inner=(1,Np_F)),Nfp_P,Np_F,K)
for j = 1:Np_F
    mapP[:,j,:] = mapP[:,j,:].+(j-1)*Nfp_P
end
mapP = reshape(mapP,Nfp_P,Np_F*K)

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


# TODO: refactor
ops = (Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth)
mesh = (rxJ,sxJ,ryJ,syJ,sJ,nrJ,nsJ,nxJ,nyJ,JP,JF,J,h,mapM,mapP)
param = (K,Np_P,Nfp_P,Np_F,Nq_P,Nh_P)
function rhs(Q,ops,mesh,param,compute_rhstest)

    Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth = ops
    rxJ,sxJ,ryJ,syJ,sJ,nrJ,nsJ,nxJ,nyJ,JP,JF,J,h,mapM,mapP = mesh
    K,Np_P,Nfp_P,Np_F,Nq_P,Nh_P = param
    w = [wq;wf]

    # Entropy projection
    VU = v_ufun(Q...)
    vh = (x->[Vq;Vf]*Pq*x).(VU)
    (ρ,ρu,ρv,ρw,E) = u_vfun(vh...)
    Uf = (x->x[Nq_P+1:end,:]).((ρ,ρu,ρv,ρw,E))

    # Convert to rho,u,v,beta vars
    β = betafun(ρ,ρu,ρv,ρw,E)
    Qh = (ρ,ρu./ρ,ρv./ρ,ρw./ρ,β)

    # Compute face values
    QM = (x->x[Nq_P+1:end,:]).(Qh)
    QP = (x->x[mapP]).(QM)

    # Lax Friedrichs Dissipation flux
    # TODO: Fix Jacobian. incorrect sJ?
    (ρM,ρuM,ρvM,ρwM,EM) = Uf
    ρuM_n = @. (ρuM*nxJ+ρvM*nyJ)/sJ # TODO: 3D lax-friedrichs?
    lam = abs.(wavespeed(ρM,ρuM_n,EM))
    LFc = .5*max.(lam,lam[mapP]).*sJ
    # LFc = 0 #TODO: TEST

    fSx,fSy,_ = euler_fluxes(QM,QP)
    normal_flux(fx,fy,u) = fx.*nxJ + fy.*nyJ - LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,fSy,Uf)
    rhsQ = (x->Vq*Lq*x).(flux)

    # Flux differencing
    ∇fh = [zeros(size(Qh[1])) for i in eachindex(Q)]

    # TODO: fix Jacobian
    for k = 1:K
        for nf = 1:Np_F
            j = nf+Np_F*(k-1)
            Qhj = [(x->x[i,j]).(Qh) for i = 1:Nh_P]
            Fxj = [euler_fluxes(UL,UR)[1] for UL in Qhj, UR in Qhj]
            Fyj = [euler_fluxes(UL,UR)[2] for UL in Qhj, UR in Qhj]
            for i = 1:5
                ∇fh[i][:,j] += 2*((rxJ[1,j]*Qrh_skew+sxJ[1,j]*Qsh_skew).*(x->x[i]).(Fxj)
                                 +(ryJ[1,j]*Qrh_skew+syJ[1,j]*Qsh_skew).*(x->x[i]).(Fyj))*ones(Nh_P,1)
            end
        end
    end

    for k = 1:K
        for nh = 1:Nh_P
            j_idx = (k-1)*Np_F+1:k*Np_F
            Qhi = [(x->x[nh,j]).(Qh) for j = j_idx]
            Fzi = [euler_fluxes(UL,UR)[3] for UL in Qhi, UR in Qhi]
            for i = 1:5
                # TODO: Why multiply by 2?
                ∇fh[i][nh,(k-1)*Np_F+1:k*Np_F] += 2*[wq/J;zeros(Nfp_P,1)][nh]*(Qth.*(x->x[i]).(Fzi))*ones(Np_F,1)
            end
        end
    end

    ∇f = (x->Vq*[Pq Lq]*diagm(1 ./ w)*x).(∇fh)
    rhsQ = @. -(∇f+rhsQ)

    rhstest = 0
    if compute_rhstest
        for fld in eachindex(rhsQ)
            rhstest += sum(wq.*VU[fld].*rhsQ[fld])
        end
    end

    return rhsQ,rhstest
end


xq,yq,zq = (x->reshape(x,Nq_P,Np_F*K)).((xq,yq,zq))
# All directions
println(" ")
println("======= All directions =======")
ρ_exact(x,y,z,t) = @. 1+0.2*sin(pi*(x+y+z-3/2*t))
ρ = @. 1+0.2*sin(pi*(xq+yq+zq))
u = ones(size(xq))
v = -1/2*ones(size(xq))
w = ones(size(xq))
p = ones(size(xq))
Q_exact(x,y,z,t) = (ρ_exact(x,y,z,t),u,v,w,p)

# # x direction test case
# println(" ")
# println("======= x direction =======")
# ρ_exact(x,y,z,t) = @. 2+ .5*sin(pi*(x-t))
# ρ = ρ_exact(xq,yq,zq,0)
# u = ones(size(xq))
# v = zeros(size(xq))
# w = zeros(size(xq))
# p = ones(size(xq))

# # y direction test case
# println(" ")
# println("======= y direction =======")
# ρ_exact(x,y,z,t) = @. 2+ .5*sin(pi*(y-t))
# ρ = ρ_exact(xq,yq,zq,0)
# u = zeros(size(xq))
# v = ones(size(xq))
# w = zeros(size(xq))
# p = ones(size(xq))

# # z direction test case
# println(" ")
# println("======= z direction =======")
# ρ_exact(x,y,z,t) = @. 2+ .5*sin(pi*(z-t))
# ρ = ρ_exact(xq,yq,zq,0)
# u = zeros(size(xq))
# v = zeros(size(xq))
# w = ones(size(xq))
# p = ones(size(xq))

Q = primitive_to_conservative(ρ,u,v,w,p)
Q = collect(Q)
resQ = [zeros(size(Q[1])) for _ in eachindex(Q)]
# rhs(Q,ops,mesh,param,false)
# @btime rhs(Q,ops,mesh,param,false)
@btime begin

for i = 1:Nsteps
    rhstest = 0
    for INTRK = 1:5
        compute_rhstest = INTRK==5
        rhsQ,rhstest = rhs(Q,ops,mesh,param,compute_rhstest)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")
    end
end

end # end time

rq2,sq2,wq2 = quad_nodes_2D(N_P+2)
Vq2 = vandermonde_2D(N_P,rq2,sq2)/VDM
xq2,yq2,zq2 = (x->Vq2*x).((x,y,z))
ρ = Vq2*Pq*Q[1]
ρ_ex = ρ_exact(xq2,yq2,zq2,T)
Q = (x->Vq2*Pq*x).(Q)
p = pfun(Q[1],(Q[2],Q[3],Q[4]),Q[5])
Q = (Q[1],Q[2]./Q[1],Q[3]./Q[1],Q[4]./Q[1],p)
Q_ex = Q_exact(xq2,yq2,zq2,T)

# # Check velocity and pressure
# println("Velocity (Avg by entries)")
# @show sum(Q[2]./Q[1])/size(Q[1],1)/size(Q[1],2)
# @show sum(Q[3]./Q[1])/size(Q[1],1)/size(Q[1],2)
# @show sum(Q[4]./Q[1])/size(Q[1],1)/size(Q[1],2)
# println("Pressure")
# p = pfun(Q[1],(Q[2],Q[3],Q[4]),Q[5])
# @show sum(p/size(Q[1],1))/size(Q[1],2)
# println("L2 error density")
# @show L2_err_ρ = sum(h*J*wq2.*(ρ-ρ_ex).^2)
#

L2_err = 0.0
for fld in eachindex(Q)
    global L2_err
    L2_err += sum(h*J*wq2.*(Q[fld]-Q_ex[fld]).^2)
end
println("L2err at final time T = $T is $L2_err\n")
