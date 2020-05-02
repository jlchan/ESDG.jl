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

function fS(uL, uR)
    return 1/6*(uL^2+uR^2+uL*uR)
end

# TODO: didn't work for T >= 0.4
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
u0(x,y) = @. -sin(pi*x)*sin(y)

"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P = 8;    # The order of approximation in polynomial dimension
Np_P = N_P+1;
Np_F = 16;    # The order of approximation in Fourier dimension
CFL  = 0.2;
T    = 5.0;  # End time

"Time integration Parameters"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (max(N_P,Np_F)+1)*(max(N_P,Np_F)+2)/2  # estimated trace constant for CFL
dt = CFL * 2 / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Initialize Reference Element in Fourier dimension"
h = 2*pi/Np_F
column = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)];
Ds = Toeplitz(column,column[[1;Np_F:-1:2]]);
s = LinRange(h,2*pi,Np_F)

"Initialize Reference Element in polynomial dimension"
rd = init_reference_interval(N_P);
@unpack r,rq,wq,rf,Dr,VDM,Vq,Vf,Vp,M,Pq,LIFT,nrJ = rd
wf = [1;1]
Nq_P = length(rq)
rp = LinRange(-1,1,100)
Vp = vandermonde_1D(N_P,rp)/VDM # Redefine Vp
Lq = LIFT

# Initialize Mesh
s,r = meshgrid(s,r)
Nfp = 2*Np_F
mapM = collect(1:Nfp)
mapP = vec(reshape([mapM[2:2:end] mapM[1:2:end-1]],1,Nfp))

# scale by Fourier dimension
# TODO: may confuse myself, any better way to do it?
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
# Qrh = diagm(vec([wq;wf]))*Drh
Qrh = 1/2*[Qr-Qr' Ef'*Br;
           -Br*Ef Br]
Qrh_skew = 1/2*(Qrh-Qrh')
# Qrh_2 = [Qr-1/2*Ef'*Br*Ef 1/2*Ef'*Br;
#          -1/2*Br*Ef       1/2*Br]
# Qrh_skew_2 = 1/2*[Qr-Qr' Ef'*Br;
#                   -Br*Ef Br]
Qsh = Qs # TODO: is this formulation correct? vanish at boundary?


ops = (Vq,Vf,wq,wf,nrJ,Qr,Qs,Ef,Br,Pq,Lq,Drh,Qrh,Qrh_skew,h)
function rhs(u,ops,compute_rhstest)
    Vq,Vf,wq,wf,nrJ,Qr,Qs,Ef,Br,Pq,Lq,Drh,Qrh,Qrh_skew,h = ops
    # Volume term
    uq,uf = (A->A*u).((Vq,Vf))
    uh = [uq;uf]

    ∇fh = zeros(size(uh)) #TODO: more informative name for this variable
    # Flux differencing in x direction
    Fs_r = zeros(size(uh,1),size(uh,1),size(uh,2))
    for k = 1:size(uh,2)
        Fs_r[:,:,k] = [fS(uL,uR) for uL in uh[:,k], uR in uh[:,k]]
    end
    QrF = Qrh.*Fs_r
    for k = 1:size(uh,2)
        ∇fh[:,k] += 2*QrF[:,:,k]*ones(size(uh,1),1)
    end

    # Flux differencing in y direction
    # TODO: simplify
    Fs_s = zeros(size(uh,2),size(uh,2),size(uh,1))
    for k = 1:size(uh,1)
        Fs_s[:,:,k] = [fS(uL,uR) for uL in uh[k,:], uR in uh[k,:]]
    end
    QsF = zeros(size(Fs_s))
    w = [wq;wf]
    for k = 1:size(uh,1)
        QsF[:,:,k] = w[k]*Qsh.*Fs_s[:,:,k]
    end
    for k = 1:size(uh,1)
        ∇fh[k,:] += 2*pi*QsF[:,:,k]*ones(size(uh,2),1)
    end

    # Spatial term
    ∇f = [Pq Lq]*diagm(vec(1 ./ [wq;wf]))*∇fh
    # Flux term
    uM = Vf*u
    uP = [uM[2,:] uM[1,:]]'
    uflux = diagm(0 => nrJ)*(@. fS(uM,uP)-uM*uM/2)
    rhsu = -(∇f+Lq*uflux)

    rhstest = 0
    if compute_rhstest
        rhstest = sum(diagm(wq)*Vq*(u.*rhsu))
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
