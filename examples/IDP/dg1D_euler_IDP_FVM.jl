using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./examples/EntropyStableEuler.jl/src")
using EntropyStableEuler
using EntropyStableEuler.Fluxes1D
const γ = 1.4



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



K = 200
T = 6.0

# Bl = -0.5
# Br = 0.5
Bl = 0.0
Br = 9.0
VX = LinRange(Bl,Br,K+1)
# Assume uniform intervals
m_i = (Br-Bl)/K # mass 
C = spdiagm(-1 => -1/2*ones(K-1), 1 => 1/2*ones(K-1))# stiffness
C[1,end] = -1/2
C[end,1] = 1/2

x = zeros(K,1)
for i = 1:K
    x[i] = (VX[i+1]+VX[i])/2
end
"""Initial condition"""
rho_x(x) = (x <= 0.0) ? 1.0 : 0.125
u_x(x) = 0.0
p_x(x) = (x <= 0.0) ? 1.0 : 0.1
# rho_x(x) = (x <= 0.0) ? 1.0 : 0.5
# u_x(x) = 1.0
# p_x(x) = (x <= 0.0) ? 1.0 : 1.0
rho_x(x) = (x <= 3.0) ? 1.0 : 0.001
u_x(x) = 0.0
p_x(x) = (x <= 3.0) ? 0.1 : 1e-7

rho = @. rho_x(x)
u = @. u_x(x)
p = @. p_x(x)
Q = primitive_to_conservative(rho,u,p)


function rhs_IDP(Q,K,m_i,C)
    p = pfun_nd.(Q[1],Q[2],Q[3])
    flux = zero.(Q)
    @. flux[1] = Q[2]
    @. flux[2] = Q[2]^2/Q[1]+p
    @. flux[3] = Q[3]*Q[2]/Q[1]+p*Q[2]/Q[1]

    dfdx = (zeros(K),zeros(K),zeros(K))
    for i = 1:K
        for c = 1:3
            dfdx[c][i] += C[i,mod1(i-1,K)]*(flux[c][mod1(i-1,K)]+flux[c][i])+
                          C[i,mod1(i+1,K)]*(flux[c][mod1(i+1,K)]+flux[c][i])
        end
    end

    visc = (zeros(K),zeros(K),zeros(K))
    for i = 1:K
        wavespd_curr = abs(wavespeed_1D(Q[1][i],Q[2][i],Q[3][i]))
        wavespd_R = abs(wavespeed_1D(Q[1][mod1(i+1,K)],Q[2][mod1(i+1,K)],Q[3][mod1(i+1,K)]))
        wavespd_L = abs(wavespeed_1D(Q[1][mod1(i-1,K)],Q[2][mod1(i-1,K)],Q[3][mod1(i-1,K)]))
        wavespd_curr_m = abs(wavespeed_1D(Q[1][i],-Q[2][i],Q[3][i]))
        wavespd_R_m = abs(wavespeed_1D(Q[1][mod1(i+1,K)],-Q[2][mod1(i+1,K)],Q[3][mod1(i+1,K)]))
        wavespd_L_m = abs(wavespeed_1D(Q[1][mod1(i-1,K)],-Q[2][mod1(i-1,K)],Q[3][mod1(i-1,K)]))
        dL = 1/2*max(wavespd_curr,wavespd_L)
        dR = 1/2*max(wavespd_curr,wavespd_R)
        for c = 1:3
            visc[c][i] = dL*(Q[c][mod1(i-1,K)]-Q[c][i]) + dR*(Q[c][mod1(i+1,K)]-Q[c][i])
        end
    end

    return -1/m_i.*(dfdx.-visc)
end

gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=0,markersize=1)
plt = plot()

dt = 0.0001
t = 0.0
Nsteps = Int(T/dt)
Q = collect(Q)
@gif for i = 1:Nsteps
    rhsQ = rhs_IDP(Q,K,m_i,C)
    global @. Q =  Q + dt*rhsQ

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")

        scatter(x,Q[1])
        #for k = 1:K
            #xp = LinRange(VX[k], VX[k+1], 3)
            #scatter!(xp,Q[1][k]*ones(length(xp)))
        #end
    end
end every 1000


# p = pfun_nd.(Q[1],Q[2],Q[3])
# for k = 1:K
#     xp = LinRange(VX[k], VX[k+1], 3)
#     scatter!(xp,Q[1][k]*ones(length(xp)))

# end
# plt