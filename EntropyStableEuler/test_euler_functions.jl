push!(LOAD_PATH, "./EntropyStableEuler")
using EntropyStableEuler

γ = 1.4

rho = 2.1
u = .1
v = -.2
p = 3

rhou = @. rho*u
rhov = @. rho*v
E = @. p/(γ-1) + .5*rho*(u^2+v^2)

U = (rho,rhou,rhov,E)

# test entropy vars
V = v_ufun(U...)
UV = u_vfun(V...)
uv_err = maximum(@. abs(UV-U))
print("uv_err = ",uv_err,"\n")

# test flux function consistency
beta = betafun(rho,u,v,E)
Fx,Fy = euler_fluxes((rho,u,v,beta),(rho,u,v,beta))
ex = abs(rho*u - Fx[1]) + abs(rho*u^2 + p - Fx[2]) + abs(rho*u*v - Fx[3]) + abs(u*(E+p) - Fx[4])
ey = abs(rho*v - Fy[1]) + abs(rho*u*v - Fy[2]) + abs(rho*v^2+p - Fy[3]) + abs(v*(E+p) - Fy[4])
print("flux consistency err = ",ex+ey,"\n")

# test entropy conservation property
rhoR = rho + rand()
ER = p/(γ-1) + .5*rhoR*(u^2+v^2)
betaR = betafun(rhoR,u,v,ER)
UL = (rho,rho*u,rho*v,E)
UR = (rhoR,rhoR*u,rhoR*v,ER)
QL = (rho,u,v,beta)
QR = (rhoR,u,v,betaR)
VL = v_ufun(UL...)
VR = v_ufun(UR...)
Fx,Fy = euler_fluxes(QL,QR)
function psix(rho,rhou,rhov,E)
    return (γ-1)*rhou
end
function psiy(rho,rhou,rhov,E)
    return (γ-1)*rhov
end

dV = [VL[fld]-VR[fld] for fld in eachindex(VL)]
ftestx = sum([dV[fld]*Fx[fld] for fld in eachindex(Fx)])
dpsix = psix(UL...)-psix(UR...)

ftesty = sum([dV[fld]*Fy[fld] for fld in eachindex(Fy)])
dpsiy = psiy(UL...)-psiy(UR...)

print("flux conservation err = ",abs(ftestx-dpsix)+abs(ftesty-dpsiy),"\n")

Fx1,Fy1 = euler_fluxes(QL,QR)
Fx2,Fy2 = euler_fluxes(QR,QL)
print("sym err = ",sum(@. abs(Fx1-Fx2) + abs(Fy1-Fy2)),"\n")
