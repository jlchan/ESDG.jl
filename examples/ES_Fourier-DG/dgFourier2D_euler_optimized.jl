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


"Constants"
const sp_tol = 1e-12


"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P = 2;    # The order of approximation in polynomial dimension
Np_P = N_P+1;
Np_F = 16;    # The order of approximation in Fourier dimension
K    = 30;   # Number of elements in polynomial dimension
CFL  = 2.5;
T    = 10.0;  # End time

"Time integration Parameters"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N_P+1)*(N_P+2)/2  # estimated trace constant for CFL
dt = CFL * 2 / CN / min(K,Np_F)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps


"Initialize Reference Element in Fourier dimension"
h = 2*pi/Np_F
column = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)]
column2 = [-pi^2/3/h^2-1/6; -((-1).^(1:Np_F-1)./(2*(sin.((1:Np_F-1)*h/2)).^2))]
Ds = Toeplitz(column,column[[1;Np_F:-1:2]])
D2s = Toeplitz(column2,column2[[1;Np_F:-1:2]])
s = LinRange(h,2*pi,Np_F)

"Mesh related variables"
VX = repeat(LinRange(0,15,K+1),2)
VY = [(-5+10/Np_F)*ones(K+1,1);5*ones(K+1,1)]
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
ys = 10/2/pi*ones(size(xr)) # TODO: hard coded
JP = xr # TODO: redundant, for clarity
JF = ys
J = @. JP*JF

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
JP = 15/K/2
JF = 5/pi
J = JF*JP
wq = J*wq
wf = JF*wf
Lq = 1/JP*Lq
Qrh = JF*Qrh
Qsh = JP*Qsh
Qrh_skew = 1/2*(Qrh-Qrh')


function flux_x!(K,Nh_P,Np_F,Qrh_skew,Qh,∇fh,Qlog)
    for k = 1:K
        for nf = 1:Np_F
            j = nf+Np_F*(k-1)
            Qhj = [(x->x[i,j]).(Qh) for i = 1:Nh_P]
            Qlogj = [(x->x[i,j]).(Qlog) for i = 1:Nh_P]
            # Fj = [euler_fluxes(UL,UR)[1] for UL in Qhj, UR in Qhj]
            Fj = fill((0.0,0.0,0.0,0.0),Nh_P,Nh_P) # TODO: hardcoded
            for ul in 1:Nh_P
                for ur in 1:Nh_P
                    Fj[ul,ur] = euler_fluxes(Qhj[ul],Qhj[ur],Qlogj[ul],Qlogj[ur])[1]
                end
            end
            update∇fh_x!(∇fh,Qrh_skew,Fj,Nh_P,j)
        end
    end
end

function update∇fh_x!(∇fh,Qrh_skew,Fj,Nh_P,j)
    for i = 1:4
        tmp = 2*Qrh_skew.*(x->x[i]).(Fj)*ones(Nh_P,1)
        for idx = 1:Nh_P
            ∇fh[i][idx,j] += tmp[idx]
        end
    end
end

function flux_y!(K,Nq_P,Np_F,Qsh,wq,Qh,∇fh,Qlog)
    # TODO: 4 - number of components hardcoded
    for k = 1:K
        j_idx = (k-1)*Np_F+1:k*Np_F
        for nh = 1:Nq_P
            Qhi = [(x->x[nh,j]).(Qh) for j = j_idx]
            Qlogi = [(x->x[nh,j]).(Qlog) for j = j_idx]
            # Fi = [euler_fluxes(UL,UR)[2] for UL in Qhi, UR in Qhi]
            Fi = fill((0.0,0.0,0.0,0.0),Np_F,Np_F) # TODO: hardcoded
            for ul in 1:Np_F
                for ur in 1:Np_F
                    Fi[ul,ur] = euler_fluxes(Qhi[ul],Qhi[ur],Qlogi[ul],Qlogi[ur])[2]
                end
            end
            update∇fh_y!(∇fh,Qsh,Fi,Np_F,nh,wq,J,k,j_idx)
        end
    end
end
# TODO: clean up function arguments
function update∇fh_y!(∇fh,Qsh,Fi,Np_F,nh,wq,J,k,j_idx)
    for i = 1:4
        tmp = 2/J*wq[nh]*(Qsh.*(x->x[i]).(Fi))*ones(Np_F,1)
        nj = 1
        for j = j_idx
            ∇fh[i][nh,j] += tmp[nj]
            nj = nj+1
        end
    end
end


# TODO: refactor
ops = (Vq,Vf,wq,wf,Pq,Lq,Qrh,Qsh,Qrh_skew)
mesh = (nrJ,JP,JF,J,h,mapM,mapP)
param = (K,Np_P,Np_F,Nq_P,Nh_P)
function rhs(Q,ops,mesh,param,compute_rhstest)

    Vq,Vf,wq,wf,Pq,Lq,Qrh,Qsh,Qrh_skew = ops
    nrJ,JP,JF,J,h,mapM,mapP = mesh
    K,Np_P,Np_F,Nq_P,Nh_P = param
    w = [wq;wf]

    # Entropy projection
    VU = v_ufun(Q...)
    vh = (x->[Vq;Vf]*Pq*x).(VU)
    (ρ,ρu,ρv,E) = u_vfun(vh...)
    Uf = (x->x[Nq_P+1:end,:]).((ρ,ρu,ρv,E))

    # Convert to rho,u,v,beta vars
    β = betafun(ρ,ρu,ρv,E)
    Qh = (ρ,ρu./ρ,ρv./ρ,β)

    # Compute face values
    QM = (x->x[Nq_P+1:end,:]).(Qh)
    QP = (x->x[mapP]).(QM)

    # Lax Friedrichs Dissipation flux
    # TODO: Fix Jacobian. incorrect sJ?
    (ρM,ρuM,ρvM,EM) = Uf
    sJ = JF
    ρuM_n = @. (nrJ*ρuM)/sJ
    lam = abs.(wavespeed(ρM,ρuM_n,EM))
    LFc = .5*max.(lam,lam[mapP])*sJ

    fSx, _ = euler_fluxes(QM,QP)
    normal_flux(fx,u) = diagm(nrJ)*fx-LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,Uf)
    rhsQ = (x->Vq*Lq*x).(flux)

    # Flux differencing
    Qlog = (log.(ρ),log.(β))
    ∇fh = [zeros(size(Qh[1])) for i in eachindex(Q)]
    flux_x!(K,Nh_P,Np_F,Qrh_skew,Qh,∇fh,Qlog)
    flux_y!(K,Nq_P,Np_F,Qsh,wq,Qh,∇fh,Qlog)
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


# Time stepping
ρ,u,v,p = vortex(Vq*reshape(x,size(x,1),size(x,2)*size(x,3)),Vq*reshape(y,size(y,1),size(y,2)*size(y,3)),0)
Q = primitive_to_conservative(ρ,u,v,p)
Q = collect(Q)
resQ = [zeros(size(Q[1])) for _ in eachindex(Q)]

# rhsQ,rhstest = rhs(Q,ops,mesh,param,false)
# @btime rhs(Q,ops,mesh,param,false)

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


# Plotting
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=3)
plot()

ρ = reshape(Pq*Q[1],Np_P,Np_F,K)
yp = LinRange(-5+10/Np_F,5,200)
sp = LinRange(h,2*pi,200)
VDM_F = vandermonde_Sinc(h,sp)
V2 = vandermonde_1D(1,sp)/vandermonde_1D(1,LinRange(h,2*pi,Np_F))
Vp = vandermonde_1D(N_P,LinRange(-1,1,10))/VDM
x_plot = mapslices((x->Vp*x*V2'),x,dims=(1,2))
y_plot = mapslices((x->Vp*x*V2'),y,dims=(1,2))
ρ_plot = mapslices((x->Vp*x*VDM_F'),ρ,dims=(1,2))
plt = scatter(vec(x_plot),vec(y_plot),vec(ρ_plot),zcolor=vec(ρ_plot),camera=(0,90))
display(plt)

println("Maximum density:")
println(maximum(Q[1]))
println("minimum density:")
println(minimum(Q[1]))
