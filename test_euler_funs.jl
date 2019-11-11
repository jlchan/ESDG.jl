push!(LOAD_PATH, "./src")
push!(LOAD_PATH, "./EntropyStableEuler")

# "Packages"
using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using Utils
using EntropyStableEuler

Np = 8;
K = 10;
rho =  2 .+ rand(Np,K)
u   = .5 .* randn(Np,K)
v   = .5 .* randn(Np,K)
w   = .5 .* randn(Np,K)
p   =  2 .+ rand(Np,K)

Q = primitive_to_conservative(rho,(u,v,w),p)

# VU = v_ufun(Q...)
# Uf = u_vfun((x->Ef*x).(VU)...)
# (rho,rhou,rhov,rhow,E) = vcat.(Q,Uf)
(rho,rhou,rhov,rhow,E) = Q
beta = betafun(rho,rhou,rhov,rhow,E)
Q = (rho,rhou./rho,rhov./rho,rhow./rho,beta) # redefine Q = (rho,u,v,β)

"test accuracy"
Fx,Fy,Fz = euler_fluxes(Q,Q)
fx = (rho.*u, rho.*u.^2 + p, rho.*u.*v, rho.*u.*w, u.*(E+p))
fy = (rho.*v, rho.*u.*v, rho.*v.^2 + p, rho.*v.*w, v.*(E+p))
fz = (rho.*w, rho.*u.*w, rho.*v.*w, rho.*w.^2 + p, w.*(E+p))
@show maximum.((x->abs.(x[:])).(Fx.-fx))
@show maximum.((x->abs.(x[:])).(Fy.-fy))
@show maximum.((x->abs.(x[:])).(Fz.-fz))

rhoL,rhoR = 2 .+ rand(2)
uL,uR = .1*randn(2)
vL,vR = .1*randn(2)
wL,wR = .1*randn(2)
pL,pR = 2 .+ rand(2)

"2d test"
UL = primitive_to_conservative(rhoL,(uL,vL),pL)
UR = primitive_to_conservative(rhoR,(uR,vR),pR)
QL = (rhoL,uL,vL,betafun(UL...))
QR = (rhoR,uR,vR,betafun(UR...))
Fx,Fy = euler_fluxes(QL,QR)
VL = v_ufun(UL...)
VR = v_ufun(UR...)
dV = VL .- VR

γ = 1.4
@show (γ-1)*(rhoL*uL - rhoR*uR)
@show sum(dV.*Fx)
@show sum(dV.*Fy)

"3d test"
UL = primitive_to_conservative(rhoL,(uL,vL,wL),pL)
UR = primitive_to_conservative(rhoR,(uR,vR,wR),pR)
QL = (rhoL,uL,vL,wL,betafun(UL...))
QR = (rhoR,uR,vR,wR,betafun(UR...))
Fx,Fy,Fz = euler_fluxes(QL,QR)
VL = v_ufun(UL...)
VR = v_ufun(UR...)
dV = VL .- VR

@show (γ-1)*(rhoL*uL - rhoR*uR)
@show sum(dV.*Fx)
@show (γ-1)*(rhoL*vL - rhoR*vR)
@show sum(dV.*Fy)
@show (γ-1)*(rhoL*wL - rhoR*wR)
@show sum(dV.*Fz)
