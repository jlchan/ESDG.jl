

using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using SparseArrays

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 7 # The order of approximation
K   = 15
CFL = .05
T   = 0.3

# viscosity, wave speed
ϵ   = .0
a   = 1

"Mesh related variables"
VX = LinRange(-1,1,K+1)
EToV = repeat([0 1],K,1) + repeat(1:K,1,2)

"Construct matrices on reference elements"
r,w = gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, r)
Dr = grad_vandermonde_1D(N, r)/V
M = inv(V*V')

"Nodes on faces, and face node coordinate"
wf = [1;1]
Vf = vandermonde_1D(N,[-1;1])/V
LIFT = M\(transpose(Vf)*diagm(wf)) # lift matrix

"Construct global coordinates"
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
x = V1*VX[transpose(EToV)]

"Connectivity maps"
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

"Make maps periodic"
mapP[1] = mapM[end]
mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1

"Quadrature operators"
rq,wq = gauss_quad(0,0,2*N)
rq,wq = gauss_lobatto_quad(0,0,N)
Vq = vandermonde_1D(N,rq)/V # vandermonde_1D(N,rq) * inv(V)
M = Vq'*diagm(wq)*Vq
LIFT = M\transpose(Vf)
Pq = (Vq'*diagm(wq)*Vq)\(Vq'*diagm(wq))

# Hybridized operators
Ef = Vf*Pq
nrJ = [-1;1]
Br = diagm(wf.*nrJ)
Qr = Pq'*M*Dr*Pq
Qrh = 1/2*[Qr-Qr' Ef'*Br;
           -Br*Ef Br]

Q_low = spdiagm(-1 => ones(N), 1 => ones(N))

"=========== done with mesh setup here ============ "

"pack arguments into tuples"
ops = (Dr,LIFT,Vf,Vq,Pq,Q_low)
vgeo = (rxJ,J)
fgeo = (nxJ,)

function burgers_exact_sol(u0,x,T,dt)
    Nsteps = ceil(Int,T/dt)
    dt = T/Nsteps
    u = u0(x)
    for i = 1:Nsteps
        t = i*dt
        u = @. u0(x-u*t)
    end
    return u
end

function rhs_nodal(u,ops,vgeo,fgeo,mapP)
    # unpack arguments
    Dr,LIFT,Vf,Vq,Pq,Q_low = ops
    rxJ,J = vgeo
    nxJ, = fgeo

    # construct sigma
    uf = Vf*u
    du = uf[mapP]-uf
    σxflux = @. .5*du*nxJ
    dudx = rxJ.*(Dr*u)
    σx = (dudx + LIFT*σxflux)./J

    # define viscosity, wavespeed parameters
    ϵ = 0.05
    tau = 1

    # compute dσ/dx
    σxf = Vf*σx
    σxP = σxf[mapP]
    σflux = @. .5*((σxP-σxf)*nxJ + tau*du)
    dσxdx = rxJ.*(Dr*σx)
    rhsσ = dσxdx + LIFT*(σflux)

    # nodal collocation
    flux = @. u^2/2
    dfdx = rxJ.*(Dr*flux)
    flux_f = Vf*flux
    df = flux_f[mapP] - flux_f
    uflux = @. .5*(df*nxJ - tau*du*abs(.5*(uf[mapP]+uf))*abs(nxJ))
    rhsu = dfdx + LIFT*uflux

    # Viscosity parameter
    coeff = V\rhsu
    indicator_modal = zeros(K,1)
    for k = 1:K
        indicator_modal[k] = coeff[end,k]^2/sum(coeff[:,k].^2)
    end
    is_shock = Float64.(indicator_modal .>= 1e-4)

    # epsilon_0 = 2.0/K/N
    # s_0 = log(10,0.01/N^4)
    # s_e = (x->log(10,x)).(indicator_modal)
    # kappa = 1.0
    # epsilon_e = zeros(K,1)
    # for i = 1:K
    #     if s_e[i] < s_0 - kappa
    #         epsilon_e[i] = 0.0
    #     elseif s_e[i] <= s_0 + kappa && s_e[i] >= s_0 - kappa
    #         epsilon_e[i] = epsilon_0*(1+sin(pi*(s_e[i]-s_0)/2/kappa))
    #     else
    #         epsilon_e[i] = epsilon_0
    #     end
    # end

    # Graph viscosity
    # visc = zeros(size(u))
    # D = zeros(size(Q_low))
    # for k = 1:K
    #     eps_e = epsilon_e[k]
    #     for i = 1:size(u,1)
    #         for j = 1:size(u,1)
    #             D[i,j] = u[j,k] - u[i,k]
    #         end
    #     end
    #     visc[:,k] = (Q_low.*D)*ones(size(u,1))
    # end

    visc = zeros(size(u))
    for k = 1:K
        # visc_2[1,k] = u[2,k]-u[1,k]
        # visc_2[end,k] = u[end-1,k]-u[end,k]
        visc[1,k] = u[2,k]-2*u[1,k]+u[end-1,mod1(k-1,K)]
        visc[end,k] = u[end-1,k]-2*u[end,k]+u[2,mod1(k+1,K)]
        for i = 2:size(u,1)-1
            visc[i,k] = u[i-1,k]-2*u[i,k]+u[i+1,k]
        end
    end

    # combine advection and viscous terms
    #rhsu = rhsu - ϵ*rhsσ
    #rhsu = rhsu - ϵ*rhsσ.*repeat(is_shock',size(rhsσ,1))
    rhsu = rhsu - visc.*repeat(is_shock',size(rhsσ,1))
    #rhsu = rhsu - visc

    return -rhsu./J
end

function fS(uL, uR)
    return 1/6*(uL^2+uR^2+uL*uR)
end

"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs_ES(u,Vq,Vf,wq,wf,K,rxJ,Qrh,Pq,LIFT,mapM,mapP,nrJ,J)
    uq,uf = (A->A*u).((Vq,Vf))
    uh = [uq;uf]
    w = [wq;wf]

    # Spatial term
    dfdx = zeros(size(uh))
    for k = 1:K
        dfdx[:,k] += 2*(rxJ*Qrh.*[fS(uL,uR) for uL in uh[:,k], uR in uh[:,k]])*ones(size(uh,1),1)
    end

    dfdx = [Pq LIFT]*diagm(1 ./ w)*dfdx

    # Flux term
    uM = uf[mapM]
    uP = uf[mapP]
    LF = @. max(abs(uP),abs(uM))*(uP-uM)
    uflux = diagm(nrJ)*(@. fS(uP,uM)-uM*uM/2)-LF
    rhsu = -(dfdx+LIFT*uflux)./J

    rhstest = 0
    #if compute_rhstest
        rhstest += sum(diagm(wq)*uq.*(Vq*rhsu))
    #end

    return rhsu, rhstest
end


"Low storage Runge-Kutta time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant
dt = CFL * 2 / (CN*K)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Perform time-stepping"
# u0(x) = @. exp(-100*(x+.5)^2)
u0(x) = @. -sin(3*pi*(x+0.1))
u = u0(x)

ulims = (minimum(u)-.5,maximum(u)+.5)

# filter_weights = ones(N+1)
# # filter_weights[end-4] = .8
# # filter_weights[end-3] = .6
# filter_weights[end-2] = .4
# filter_weights[end-1] = .1
# filter_weights[end] = .0
# Filter = V*(diagm(filter_weights)/V)

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,100))/V
gr(aspect_ratio=0.5,legend=false,
   markerstrokewidth=0,markersize=2)
# plot()

wJq = diagm(wq)*(Vq*J)

resu = zeros(size(x)) # Storage for the Runge kutta residual storage
energy = zeros(Nsteps)
interval = 25
for i = 1:Nsteps
    for INTRK = 1:5
        rhsu = rhs_nodal(u,ops,vgeo,fgeo,mapP)
        #rhsu, rhstest = rhs_ES(u,Vq,Vf,wq,wf,K,rxJ,Qrh,Pq,LIFT,mapM,mapP,nrJ,J)
        # rhsu .= (Filter*rhsu)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu

        # u .= (Filter*u)
    end
    energy[i] = sum(sum(wJq.*(Vq*u).^2))

    if i%interval==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        # plot(Vp*x,Vp*u,ylims=ulims,title="Timestep $i out of $Nsteps",lw=2)
        # scatter!(x,u,xlims=(-1,1),ylims=ulims)
    end
end

scatter(x,u,markersize=1,xlims=(-1,1),ylims=(-2,2)) # plot nodal values
display(plot!(Vp*x,Vp*u)) # plot interpolated solution at fine points

coeff = V\u
indicator_modal = zeros(K)
for k = 1:K
    #indicator_modal[k] = norm(u[:,k]-Pq*u[:,k])
    indicator_modal[k] = coeff[end,k]^2/sum(coeff[:,k].^2)
end

using DelimitedFiles

open("indicator_modal.txt","w") do io
    writedlm(io,indicator_modal)
end
