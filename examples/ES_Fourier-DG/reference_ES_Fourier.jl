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
# u0(x,y) = @. -sin(pi*x)*sin(y)
u0(x,y) = @. exp(-1*((y-pi)^2+2*pi*x^2))

"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P = 8;    # The order of approximation in polynomial dimension
Np_P = N_P+1;
Np_F = 16;    # The order of approximation in Fourier dimension
CFL  = 0.2;
T    = 1.2;  # End time

"Time integration Parameters"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (max(N_P,Np_F)+1)*(max(N_P,Np_F)+2)/2  # estimated trace constant for CFL
dt = CFL / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Initialize Reference Element in Fourier dimension"
h = 2*pi/Np_F
column = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)]
column2 = [-pi^2/3/h^2-1/6; -((-1).^(1:Np_F-1)./(2*(sin.((1:Np_F-1)*h/2)).^2))]
Ds = Toeplitz(column,column[[1;Np_F:-1:2]])
D2s = Toeplitz(column2,column2[[1;Np_F:-1:2]])
s = LinRange(h,2*pi,Np_F)

"Initialize Reference Element in polynomial dimension"
rd = init_reference_interval(N_P);
@unpack r,rq,wq,rf,Dr,VDM,Vq,Vf,Vp,M,Pq,LIFT,nrJ = rd
wf = [1;1]
Nq_P = length(rq)
rp = LinRange(-1,1,100)
Vp = vandermonde_1D(N_P,rp)/VDM # Redefine Vp
Lq = LIFT

"Initialize Mesh"
s,r = meshgrid(s,r)
Nfp = 2*Np_F

"scale by Fourier dimension"
M = h*M
wq = h*wq
wf = h*wf

"Hybridized operators"
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

ops = (Vq,Vf,wq,wf,Pq,Lq,Qrh,Qsh,D2s,nrJ,h)
function rhs(u,ops,compute_rhstest)
    Vq,Vf,wq,wf,Pq,Lq,Qrh,Qsh,D2s,nrJ,h = ops
    # Volume term
    uq,uf = (A->A*u).((Vq,Vf))
    uh = [uq;uf]
    w = [wq;wf]
    ∇fh = zeros(size(uh))

    # Flux differencing in x direction
    for k = 1:size(uh,2)
        ∇fh[:,k] += 2*Qrh.*[fS(uL,uR) for uL in uh[:,k], uR in uh[:,k]]*ones(size(uh,1),1)
    end

    # Flux differencing in y direction
    for k = 1:size(uh,1)
        ∇fh[k,:] += 2*pi*w[k]*Qsh.*[fS(uL,uR) for uL in uh[k,:], uR in uh[k,:]]*ones(size(uh,2),1)
    end

    # Spatial term
    ∇f = [Pq Lq]*diagm(1 ./ w)*∇fh

    # Flux term
    uM = Vf*u
    uP = [uM[2,:] uM[1,:]]'
    uflux = diagm(nrJ)*(@. fS(uM,uP)-uM*uM/2)

    # Artificial Viscosity
    ϵ = 0.05
    Δf = ϵ*u*D2s'
    rhsu = -(∇f+Lq*uflux-Δf)

    rhstest = 0
    if compute_rhstest
        rhstest += sum(diagm(wq)*uq.*(Vq*rhsu))
    end

    return rhsu,rhstest
end

u = @. u0(r,s)

# Test single element
resu = zeros(size(u))
for i = 1:Nsteps
    rhstest = 0
    for INTRK = 1:5
        compute_rhstest = INTRK==5
        rhsu,rhstest = rhs(u,ops,compute_rhstest)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u += rk4b[INTRK]*resu
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")
    end
end


# Plotting
gr(aspect_ratio=1/pi, legend=false,
   markerstrokewidth=0, markersize=2.5,
   xlims=(-1,1),ylims=(h,2*pi),showaxis=false)
plot()

sp = LinRange(h,2*pi,100)
VDM_F = vandermonde_Sinc(h,sp)
V2 = vandermonde_1D(1,sp)/vandermonde_1D(1,LinRange(h,2*pi,Np_F))

plt = scatter(Vp*r*V2',Vp*s*V2',Vp*u*VDM_F',zcolor=Vp*u*VDM_F',camera=(0,90))
display(plt)
# exact_sol = burgers_exact_sol_2D(u0,Vp*r*V2',Vp*s*V2',T,dt/100)
# scatter(Vp*r*V2',Vp*s*V2',exact_sol,zcolor=exact_sol,camera=(0,90))

# L2 error
if compute_L2_err
    rq2,wq2 = gauss_quad(0,0,N_P+2)
    wq2 = wq2*h
    Vq2 = vandermonde_1D(N_P,rq2)/VDM
    xq2 = Vq2*r
    yq2 = Vq2*s

    exact_sol = burgers_exact_sol_2D(u0,xq2,yq2,T,dt/100)
    L2_err = sum(diagm(wq2)*(Vq2*u-exact_sol).^2)
    @show L2_err
end
