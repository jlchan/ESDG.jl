using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack


push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
using EntropyStableEuler.Fluxes1D
const γ = 1.4#5/3# 1.4

function wavespeed_1D(rho,rhou,E)
    p = pfun_nd(rho,(rhou,),E)
    cvel = @. sqrt(γ*p/rho)
    return @. abs(rhou/rho) + cvel
end
unorm(U) = sum(map((x->x.^2),U))
function pfun_nd(rho, rhoU, E)
    rhoUnorm2 = unorm(rhoU)./rho
    return @. (γ-1)*(E - .5*rhoUnorm2)
end

function primitive_to_conservative_hardcode(rho,U,p)
    rhoU = rho.*U
    Unorm = unorm(U)
    E = @. p/(γ-1) + .5*rho*Unorm
    return (rho,rhoU,E)
end


"Approximation parameters"
N = 1 # The order of approximation
K = 50
T = 0.2 # endtime
#T = 6.0
#T = 0.0001

# Sod shocktube
const Bl = -0.5
const Br = 0.5
const rhoL = 1.0
const rhoR = 0.125
const pL = 1.0
const pR = 0.1
const xC = 0.0

# # Leblanc shocktube
# const Bl = 0.0
# const Br = 9.0
# const rhoL = 1.0
# const rhoR = 0.001
# const pL = 0.1
# const pR = 1e-7
# const xC = 3.0

"Mesh related variables"
#VX = LinRange(-0.5,0.5,K+1)
VX = LinRange(Bl,Br,K+1)
EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K))

"Initialize reference element"
r,_ = gauss_lobatto_quad(0,0,N)   # Reference nodes
VDM = vandermonde_1D(N,r)         # modal to nodal
Dr = grad_vandermonde_1D(N,r)/VDM # nodal differentiation
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1]) # nodal linear interpolation

Nq = N+1
rq,wq = gauss_quad(0,0,Nq)
Vq = vandermonde_1D(N,rq)/VDM
M = Vq'*diagm(wq)*Vq
Mlump = zeros(size(M))
Mlump_inv = zeros(size(M))
for i = 1:N+1
    Mlump[i,i] = sum(M[i,:])
    Mlump_inv[i,i] = 1.0/Mlump[i,i]
end
S = M*Dr
L = zeros(N+1,2)
L[1,1] = -1.0/2.0
L[end,end] = 1.0/2.0

rf = [-1.0;1.0]
nrJ = [-1.0;1.0]
Vf = vandermonde_1D(N,rf)/VDM

"""Connectivity map"""
x = V1*VX[transpose(EToV)]
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

# """Periodic"""
# mapP[1] = mapM[end]
# mapP[end] = mapP[1]

"""Geometric factors"""
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1.0

"""Initial condition"""
# rho_x(x) = (x <= 0.0) ? 1.0 : 0.125
# u_x(x) = 0.0
# p_x(x) = (x <= 0.0) ? 1.0 : 0.1
# rho_x(x) = (x <= 3.0) ? 1.0 : 0.001
# u_x(x) = 0.0
# p_x(x) = (x <= 3.0) ? 0.1 : 1e-7
rho_x(x) = (x <= xC) ? rhoL : rhoR
u_x(x) = 0.0
p_x(x) = (x <= xC) ? pL : pR 
# rho_x(x) = (x <= 0.0) ? 1.0 : 0.8
# u_x(x) = 0.0
# p_x(x) = (x <= 0.0) ? 1.0 : 0.8
# rho_x(x) = (x <= 0.0) ? 1.0 : 0.125
# u_x(x) = 1.0
# p_x(x) = (x <= 0.0) ? 1.0 : 1.0


rho = @. rho_x(x)
u = @. u_x(x)
p = @. p_x(x)
Q = primitive_to_conservative_hardcode(rho,u,p)

# hardcoded first order IDP
function rhs_IDPlow(Q,K,N)
    p = pfun_nd.(Q[1],Q[2],Q[3])
    flux = zero.(Q)
    @. flux[1] = Q[2]
    @. flux[2] = Q[2]^2/Q[1]+p
    @. flux[3] = Q[3]*Q[2]/Q[1]+p*Q[2]/Q[1]


    """Periodic case"""    
    # J = 1/K/2 # assume uniform interval
    # dfdx = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    # for i = 1:K*(N+1)
    #     for c = 1:3
    #         dfdx[c][i] = 1/2*(flux[c][mod1(i+1,K*(N+1))] - flux[c][mod1(i-1,K*(N+1))])
    #     end
    # end
     
    # visc = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    # for i = 1:K*(N+1)
    #     wavespd_curr = abs(wavespeed_1D(Q[1][i],Q[2][i],Q[3][i]))
    #     wavespd_R = abs(wavespeed_1D(Q[1][mod1(i+1,K*(N+1))],Q[2][mod1(i+1,K*(N+1))],Q[3][mod1(i+1,K*(N+1))]))
    #     wavespd_L = abs(wavespeed_1D(Q[1][mod1(i-1,K*(N+1))],Q[2][mod1(i-1,K*(N+1))],Q[3][mod1(i-1,K*(N+1))]))
    #     wavespd_curr_m = abs(wavespeed_1D(Q[1][i],-Q[2][i],Q[3][i]))
    #     wavespd_R_m = abs(wavespeed_1D(Q[1][mod1(i+1,K*(N+1))],-Q[2][mod1(i+1,K*(N+1))],Q[3][mod1(i+1,K*(N+1))]))
    #     wavespd_L_m = abs(wavespeed_1D(Q[1][mod1(i-1,K*(N+1))],-Q[2][mod1(i-1,K*(N+1))],Q[3][mod1(i-1,K*(N+1))]))
    #     dL = 1/2*max(wavespd_curr,wavespd_L)
    #     dR = 1/2*max(wavespd_curr,wavespd_R)
    #     for c = 1:3
    #         visc[c][i] = dL*(Q[c][mod1(i-1,K*(N+1))]-Q[c][i]) + dR*(Q[c][mod1(i+1,K*(N+1))]-Q[c][i])
    #     end
    # end 

    # rhsQ = @. 1/J*(-dfdx+visc)
    # return rhsQ
    
    """shocktube"""
    J = (Br-Bl)/K/2 # assume uniform interval
    dfdx = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    for i = 2:K*(N+1)-1
        for c = 1:3
            dfdx[c][i] = 1/2*(flux[c][mod1(i+1,K*(N+1))] - flux[c][mod1(i-1,K*(N+1))])
        end
    end
    dfdx[1][1] = 1/2*(flux[1][2] - 0.0)
    dfdx[2][1] = 1/2*(flux[2][2] - pL)
    dfdx[3][1] = 1/2*(flux[3][2] - 0.0)
    dfdx[1][end] = 1/2*(0.0 - flux[1][end-1])
    dfdx[2][end] = 1/2*(pR - flux[2][end-1])
    dfdx[3][end] = 1/2*(0.0 - flux[3][end-1])

    visc = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    for i = 2:K*(N+1)-1
        wavespd_curr = wavespeed_1D(Q[1][i],Q[2][i],Q[3][i])
        wavespd_R = wavespeed_1D(Q[1][mod1(i+1,K*(N+1))],Q[2][mod1(i+1,K*(N+1))],Q[3][mod1(i+1,K*(N+1))])
        wavespd_L = wavespeed_1D(Q[1][mod1(i-1,K*(N+1))],Q[2][mod1(i-1,K*(N+1))],Q[3][mod1(i-1,K*(N+1))])
        dL = 1/2*max(wavespd_curr,wavespd_L)
        dR = 1/2*max(wavespd_curr,wavespd_R)
        for c = 1:3
            visc[c][i] = dL*(Q[c][mod1(i-1,K*(N+1))]-Q[c][i]) + dR*(Q[c][mod1(i+1,K*(N+1))]-Q[c][i])
        end
    end

    # i = 1, (rho, rhoU, E) = (1.0, 0.0, 0.1/(γ-1))
    wavespd_curr = wavespeed_1D(Q[1][1],Q[2][1],Q[3][1])
    wavespd_R = wavespeed_1D(Q[1][2],Q[2][2],Q[3][2])
    wavespd_L = wavespeed_1D(rhoL,0.0,pL/(γ-1))
    dL = 1/2*max(wavespd_curr,wavespd_L)
    dR = 1/2*max(wavespd_curr,wavespd_R)
    visc[1][1] = dL*(rhoL-Q[1][1]) + dR*(Q[1][2]-Q[1][1])
    visc[2][1] = dL*(0.0-Q[2][1]) + dR*(Q[2][2]-Q[2][1])
    visc[3][1] = dL*(pL/(γ-1)-Q[3][1]) + dR*(Q[3][2]-Q[3][1])


    # i = end, (rho, rhoU, E) = (1e-3, 0.0, 1e-7/(γ-1))
    wavespd_curr = wavespeed_1D(Q[1][end],Q[2][end],Q[3][end])
    wavespd_R = wavespeed_1D(rhoR,0.0,pR/(γ-1))
    wavespd_L = wavespeed_1D(Q[1][end-1],Q[2][end-1],Q[3][end-1])
    dL = 1/2*max(wavespd_curr,wavespd_L)
    dR = 1/2*max(wavespd_curr,wavespd_R)
    visc[1][end] = dL*(Q[1][end-1]-Q[1][end]) + dR*(rhoR-Q[1][end])
    visc[2][end] = dL*(Q[2][end-1]-Q[2][end]) + dR*(0.0-Q[2][end])
    visc[3][end] = dL*(Q[3][end-1]-Q[3][end]) + dR*(pR/(γ-1)-Q[3][end])
    
    rhsQ = @. 1/J*(-dfdx+visc)
    return rhsQ
end

function rhs_nodal(Q,Vf,mapP,rxJ,S,L,N,K,J,Mlump,Mlump_inv)
    p = pfun_nd(Q...)
    flux = zero.(Q)
    @. flux[1] = Q[2]
    @. flux[2] = Q[2]^2/Q[1]+p
    @. flux[3] = Q[3]*Q[2]/Q[1]+p*Q[2]/Q[1]

    # flux derivative
    dfdx = (x->rxJ*S*x).(flux)

    # jump in flux
    flux_f = (x->Vf*x).(flux)
    df = (x->x[mapP]-x).(flux_f)
    # Boundary condition (Sod tube)
    df[1][1] = -flux_f[1][1]
    df[2][1] = 1.0-flux_f[2][1]
    df[3][1] = -flux_f[3][1]
    df[1][end] = -flux_f[1][end]
    df[2][end] = 0.1-flux_f[2][end]
    df[3][end] = -flux_f[3][end]
    df = (x->L*x).(df)

    # Artificial Viscosity
    visc = (zeros(N+1,K),zeros(N+1,K),zeros(N+1,K))
    D = zeros(N+1,N+1)
    T = zeros(N+1,N+1)
    D_arr = zeros(N+1,K) # Array of d_ii
    for k = 1:K
        # Construct graph viscosity coefficients
        for i = 1:N+1
            for j = 1:N+1
                wavespd = max(wavespeed_1D(Q[1][i,k],Q[2][i,k],Q[3][i,k]),
                              wavespeed_1D(Q[1][j,k],Q[2][j,k],Q[3][j,k]))
                # Boundary nodes (hard coded)
                if i == 1 && j == 1
                    #T[i,j] = max(wavespd*abs(S[i,j]-1/2),wavespd*abs(S[i,j]-1/2))
                elseif i == N+1 && j == N+1
                    #T[i,j] = max(wavespd*abs(S[i,j]+1/2),wavespd*abs(S[i,j]+1/2))
                else
                    T[i,j] = max(wavespd*abs(S[i,j]),wavespd*abs(S[i,j]))
                    if i != j
                        D_arr[i,k] -= T[i,j]
                    end
                end
            end
        end

        for c = 1:3
            # Difference matrix
            for i = 1:N+1
                for j = 1:N+1
                    D[i,j] = Q[c][j,k]-Q[c][i,k]
                end
            end
            visc[c][:,k] = (T.*D)*ones(N+1)

            # Exterior nodes TODO: hardcoded
            wavespd = max(wavespeed_1D(Q[1][end,mod1(k-1,K)],Q[2][end,mod1(k-1,K)],Q[3][end,mod1(k-1,K)]),
                        wavespeed_1D(Q[1][1,k],Q[2][1,k],Q[3][1,k]))
            visc[c][1,k] += wavespd/2*(Q[c][end,mod1(k-1,K)]-Q[c][1,k])
            D_arr[1,k] -= wavespd/2

            wavespd = max(wavespeed_1D(Q[1][end,k],Q[2][end,k],Q[3][end,k]),
                        wavespeed_1D(Q[1][1,mod1(k+1,K)],Q[2][1,mod1(k+1,K)],Q[3][1,mod1(k+1,K)]))
            visc[c][end,k] += wavespd/2*(Q[c][1,mod1(k+1,K)]-Q[c][end,k])
            D_arr[end,k] -= wavespd/2

            # if k <= K-1
            #     wavespd = max(wavespeed_1D(Q[1][end,k],Q[2][end,k],Q[3][end,k]),
            #                 wavespeed_1D(Q[1][1,k+1],Q[2][1,k+1],Q[3][1,k+1]))
            #     visc[c][end,k] += wavespd/2*(Q[c][1,k+1]-Q[c][end,k])
            #     D_arr[end,k] -= wavespd/2
            # else
            #     # wavespd = max(wavespeed_1D(Q[1][end,k],Q[2][end,k],Q[3][end,k]),
            #     #             wavespeed_1D(0.125,0.0,0.1/(γ-1)))  
            #     # if c == 1
            #     #     visc[1][end,k] += wavespd/2*(0.125-Q[1][end,k])
            #     # elseif c == 2
            #     #     visc[2][end,k] += wavespd/2*(0.0-Q[2][end,k])
            #     # else
            #     #     visc[3][end,k] += wavespd/2*(0.1/(γ-1)-Q[3][end,k])
            #     # end
            #     # D_arr[end,k] -= wavespd/2
            # end
        end
    end

    dt = Inf
    for k = 1:K
        for i = 1:N+1
            dt = min(dt,-J[1]*Mlump[i,i]/(4*D_arr[i,k]))
        end
    end

    rhsQ = dfdx .+ df .- visc
    rhsQ = (x->-Mlump_inv*x./J).(rhsQ)
    return rhsQ, dt
end


# Time stepping
"Time integration"
t = 0.0
Q = collect(Q)
resQ = [zeros(size(x)),zeros(size(x)),zeros(size(x))]

# Forward Euler
while t < T
    #rhsQ,dt = rhs_nodal(Q,Vf,mapP,rxJ,S,L,N,K,J,Mlump,Mlump_inv)
    rhsQ = rhs_IDPlow(Q,K,N)
    dt = 0.0001
    @. Q = Q + dt*rhsQ
    global t = t + dt
    println("Current time $t with time step size $dt, and final time $T")
end

# rk4a,rk4b,rk4c = rk45_coeffs()
# Nsteps = Int(T/0.0001)
# for i = 1:1000#Nsteps
#     for INTRK = 1:5
#         rhsQ,dt = rhs_nodal(Q,Vf,mapP,rxJ,S,L,N,K,J,Mlump,Mlump_inv)
#         dt = 0.0001
#         @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
#         @. Q   += rk4b[INTRK]*resQ
#     end
#     if i%10 == 0 || i == Nsteps
#         println("Number of time steps $i out of $Nsteps")
#     end
# end
#=
# SSPRK(3,3)
for i = 1:5
    rhsQ =  rhs_nodal(Q,Vf,mapP,rxJ,S,L,N,K,J,Mlump,Mlump_inv)
    w1 = @. Q + dt*rhsQ
    rhsQ = rhs_nodal(w1,Vf,mapP,rxJ,S,L,N,K,J)
    z1 = @. w1 + dt*rhsQ
    w2 = @. 3/4*Q + 1/3*z1
    rhsQ = rhs_nodal(w2,Vf,mapP,rxJ,S,L,N,K,J)
    z2 = @. w2 + dt*rhsQ
    @. Q = 1/3*Q + 2/3*z2
    if i%10 == 0 || i == Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end
=#

Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*Q[1])

