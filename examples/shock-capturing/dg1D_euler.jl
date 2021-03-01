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
const γ = 5.0/3.0
#const γ = 1.4

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
function betafun_nd(rho,rhoU,E)
    p = pfun_nd(rho,rhoU,E)
    return (@. rho/(2*p))
end
function primitive_to_conservative_nd(rho,U,p)
    rhoU = rho.*U
    Unorm = unorm(U)
    E = @. p/(γ-1) + .5*rho*Unorm
    return (rho,rhoU,E)
end



"Approximation parameters"
N = 5 # The order of approximation
K = 20
CFL = 0.2
T   = 2.0 # endtime

# # Sod shocktube
# const xL = -0.5
# const xR = 0.5
# const rhoL = 1.0
# const rhoR = 0.125
# const pL = 1.0
# const pR = 0.1
# const xC = 0.0

# Leblanc shocktube
const xL = 0.0
const xR = 9.0
const rhoL = 1.0
const rhoR = 0.001
const pL = 0.1
const pR = 1e-7
const xC = 3.0

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+1)/2  # estimated trace constant for CFL
h  = 1/K
dt = CFL * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"Mesh related variables"
VX = LinRange(xL,xR,K+1)
EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K))


# initialize ref element and mesh
# specify lobatto nodes to automatically get DG-SEM mass lumping
# rd = init_reference_interval(N)
# @unpack r,VDM,Dr,V1,rq,wq,Vq,M,Pq,rf,nrJ,Vf,LIFT,rp,Vp = rd

# Construct matrices on reference elements
r,_ = gauss_lobatto_quad(0,0,N)
VDM = vandermonde_1D(N, r)
Dr = grad_vandermonde_1D(N, r)/VDM

V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])

#Nq = N+2
#Nq = N+1

# Collocation
Nq = N
rq,wq = gauss_lobatto_quad(0,0,Nq)

# Nq = N+1
# rq,wq = gauss_quad(0,0,Nq)

Vq = vandermonde_1D(N, rq)/VDM
M = Vq'*diagm(wq)*Vq
Pq = M\(Vq'*diagm(wq))

rf = [-1.0;1.0]
nrJ = [-1.0;1.0]
Vf = vandermonde_1D(N,rf)/VDM
LIFT = M\(Vf') # lift matrix

# plotting nodes
rp = LinRange(-1,1,50)
Vp = vandermonde_1D(N,rp)/VDM


"Construct global coordinates"
V1 = vandermonde_1D(1,r)/vandermonde_1D(1,[-1;1])
x = V1*VX[transpose(EToV)]

"Connectivity maps"
xf = Vf*x
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]

# "Make periodic"
# mapP[1] = mapM[end]
# mapP[end] = mapM[1]

"Geometric factors and surface normals"
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1

xq = Vq*x
wJq = diagm(wq)*(Vq*J)

Qr = Pq'*M*Dr*Pq
Ef = Vf*Pq
wf = [1.0;1.0]
Br = diagm(wf.*nrJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
         -Br*Ef  Br]
Qrh_skew = 0.5*(Qrh-transpose(Qrh))
Qrh_sparse = droptol!(sparse(Qrh_skew),1e-12)

Vh = droptol!(sparse([eye(length(wq)); Ef]),1e-12)
Ph = droptol!(sparse(diagm(@. 1/wq)*transpose(Vh)),1e-12)
Lf = droptol!(sparse(diagm(@. 1/wq)*(transpose(Ef)*diagm(wf))),1e-12)

# define initial conditions at quadrature nodes
rho_x(x) = (x < xC) ? rhoL : rhoR
u_x(x) = 0.0
p_x(x) = (x < xC) ? pL : pR

rho = @. rho_x(xq)
u = @. u_x(xq)
p = @. p_x(xq)#1/(γ-1)*p_x(xq)

# # nodal
# rho = @. rho_x(x)
# u = @. u_x(x)
# p = @. p_x(x)
Q = primitive_to_conservative_nd(rho,u,p)

function rhs_nodal(Q,Dr,LIFT,Vf,Pq,rxJ,J,nxJ,mapP)
    QM = (x->Vf*Pq*x).(Q)
    QP = (x->x[mapP]).(QM)
    dQ = @. QP-QM

    Q = (x->Pq*x).(Q)
    p = pfun_nd(Q...)
    flux = zero.(Q)
    @. flux[1] = Q[2]
    @. flux[2] = Q[2]^2/Q[1]+p
    @. flux[3] = Q[3]*Q[2]/Q[1]+p*Q[2]/Q[1]

    dfdx = (x->rxJ*Dr*x).(flux)

    flux_f = (x->Vf*x).(flux)
    df = (x->x[mapP]-x).(flux_f)
    tau = 1.0
    normal_flux(dQ,df,QP,QM) = @. .5*(df*nxJ - tau*dQ*abs(.5*(QP+QM))*abs(nxJ))
    Qflux = normal_flux.(dQ,df,QP,QM)

    Qflux = (x->LIFT*x).(Qflux)
    rhsQ = dfdx .+ Qflux
    rhsQ = (x->Vq*(-x./J)).(rhsQ)
    return rhsQ
end

function Roe_flux(rhoL,rhouL,EL,rhoR,rhouR,ER)
    pL = pfun_nd(rhoL,rhouL,EL)
    pR = pfun_nd(rhoR,rhouR,ER)
    HL = (pL+EL)/rhoL
    HR = (pR+ER)/rhoR
    uL = rhouL/rhoL
    uR = rhouR/rhoR
    drho = rhoR - rhoL
    dp = pR - pL
    du = uR - uL

    u_roeavg = (sqrt(rhoL)*uL+sqrt(rhoR)*uR)/(sqrt(rhoL)+sqrt(rhoR))
    H_roeavg = (sqrt(rhoL)*HL+sqrt(rhoR)*HR)/(sqrt(rhoL)+sqrt(rhoR))
    c_roeavg = (γ-1.0)*(H_roeavg-u_roeavg^2/2)
    rho_roeavg = sqrt(rhoL*rhoR)

    lambda_1 = u_roeavg - c_roeavg
    lambda_2 = u_roeavg
    lambda_3 = u_roeavg + c_roeavg

    alpha_1 = 1/(2*c_roeavg^2)*(dp-c_roeavg*rho_roeavg*du)
    alpha_2 = drho - 1/c_roeavg^2*dp
    alpha_3 = 1/(2*c_roeavg^2)*(dp+c_roeavg*rho_roeavg*du)

    return 1/2*abs(lambda_1)*alpha_1*[1; u_roeavg-c_roeavg; H_roeavg-u_roeavg*c_roeavg] +
           1/2*abs(lambda_2)*alpha_2*[1; u_roeavg; u_roeavg^2/2] +
           1/2*abs(lambda_3)*alpha_3*[1; u_roeavg+c_roeavg; H_roeavg+u_roeavg*c_roeavg]
end

"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs_ES(Q,VDM,Vq,Vf,wq,wf,Pq,mapP,LIFT,nxJ,rxJ,Qrh_skew,J,K,compute_rhstest=false)
    Nq = size(Vq,1)
    Nf = size(Vf,1)
    Nh = Nq+Nf
    w = [wq;wf]

    VU = v_ufun(Q...)

    @show "===================================="
    # TODO: when L2 project and interp at face quad, value blows up
    tmp = (x->Vf*Pq*x).(VU)
    #tmp = (x->Pq*x).(VU)
    #tmp = (x->x[[1;size(VU[1],1)],:]).(VU)
    Uf = u_vfun(tmp...)

    (rho,rhou,E) = vcat.(Q,Uf)

    beta = betafun_nd.(rho,rhou,E)
    Qh = (rho, rhou./rho, beta)
    QM = (x->x[Nq+1:Nh,:]).(Qh)
    QP = (x->x[mapP]).(QM)
    
    # Boundary condition
    QP[1][1] = rhoL
    QP[2][1] = 0.0
    QP[3][1] = betafun_nd(rhoL,0.0,pL/(γ-1))
    QP[1][end] = rhoR
    QP[2][end] = 0.0
    QP[3][end] = betafun_nd(rhoR,0.0,pR/(γ-1))
    # QP[1][1] = 1.0
    # QP[2][1] = 0.0
    # QP[3][1] = betafun(1.0,0.0,0.1/(γ-1))
    # QP[1][end] = 0.001
    # QP[2][end] = 0.0
    # QP[3][end] = betafun(0.001,0.0,1e-7/(γ-1))

    (rhoM,rhouM,EM) = Uf
    rhoUM_n = @. rhouM*nxJ
    lam = abs.(wavespeed_1D(rhoM,rhoUM_n,EM))
    #LFc = .5*max.(lam,lam[mapP])

    # Boundary condition
    lamP = lam[mapP]
    lamP[1] = abs(wavespeed_1D(rhoL,0.0,pL/(γ-1)))
    lamP[end] = abs(wavespeed_1D(rhoR,0.0,pR/(γ-1)))
    LFc = .5*max.(lam,lamP)

    fSx = euler_fluxes(QM,QP)
    #normal_flux(fx,u) = fx.*nxJ - LFc.*(u[mapP]-u)
    #flux = normal_flux.(fSx,Uf)

    # Boundary conditions
    flux = [zeros(size(nxJ)),zeros(size(nxJ)),zeros(size(nxJ))]
    UP = (x->x[mapP]).(Uf)
    UP[1][1] = rhoL
    UP[2][1] = 0.0
    UP[3][1] = pL/(γ-1)
    UP[1][end] = rhoR
    UP[2][end] = 0.0
    UP[3][end] = pR/(γ-1)
    flux[1] = fSx[1].*nxJ - LFc.*(UP[1]-Uf[1])
    flux[2] = fSx[2].*nxJ - LFc.*(UP[2]-Uf[2])
    flux[3] = fSx[3].*nxJ - LFc.*(UP[3]-Uf[3])

    rhsQ = (x->LIFT*x).(flux)

    dfdx = [zeros(size(Qh[1])) for i in eachindex(Q)]
    for k = 1:K
        Qhk = [(x->x[i,k]).(Qh) for i = 1:Nh]
        Fk = [euler_fluxes(UL,UR) for UL in Qhk, UR in Qhk]
        for i = 1:3
            dfdx[i][:,k] += 2*(rxJ*Qrh_skew.*(x->x[i]).(Fk))*ones(Nh,1)
        end
    end

    dfdx = (x->[Pq LIFT]*diagm(1 ./ w)*x).(dfdx)

    # # Viscosity
    # VU_modal = (x->Pq*x).(VU)
    # UV = u_vfun((x->Vq*x).(VU_modal)...)
    # UV_modal = (x->Pq*x).(UV)

    # coeff_1 = VDM\VU_modal[1]
    # coeff_2 = VDM\VU_modal[1]
    # coeff_3 = VDM\VU_modal[1]
    # indicator_modal_V = zeros(K)
    # for k = 1:K
    #     indicator_modal_V[k] += coeff_1[end,k]^2/sum(coeff_1[:,k].^2)
    #     indicator_modal_V[k] += coeff_2[end,k]^2/sum(coeff_2[:,k].^2)
    #     indicator_modal_V[k] += coeff_3[end,k]^2/sum(coeff_3[:,k].^2)
    #     indicator_modal_V[k] = indicator_modal_V[k]/3
    # end
    # is_shock = Float64.(indicator_modal_V .>= 1e-6)
    # #is_shock = ones(length(indicator_modal_V))

    # indicator_proj_U = zeros(K)
    # for k = 1:K
    #     indicator_proj_U[k] += norm(UV[1][:,k]-Vq*UV_modal[1][:,k])^2
    #     indicator_proj_U[k] += norm(UV[2][:,k]-Vq*UV_modal[2][:,k])^2
    #     indicator_proj_U[k] += norm(UV[3][:,k]-Vq*UV_modal[3][:,k])^2
    #     indicator_proj_U[k] = sqrt(indicator_proj_U[k])
    # end
    # is_shock = Float64.(indicator_proj_U .>= 5e-3)

    visc = [zeros(size(rhsQ[1])), zeros(size(rhsQ[1])), zeros(size(rhsQ[1]))]
    Q_modal = (x->Pq*x).(Q)
    
    # LF flux
    for i = 2:K*size(rhsQ[1],1)-1
        wavespd_curr = wavespeed_1D(Q_modal[1][i],Q_modal[2][i],Q_modal[3][i])
        wavespd_R = wavespeed_1D(Q_modal[1][i+1],Q_modal[2][i+1],Q_modal[3][i+1])
        wavespd_L = wavespeed_1D(Q_modal[1][i-1],Q_modal[2][i-1],Q_modal[3][i-1])
        dL = 1/2*max(wavespd_L,wavespd_curr)
        dR = 1/2*max(wavespd_R,wavespd_curr)
        for c = 1:3
            visc[c][i] = dL*(Q_modal[c][i-1]-Q_modal[c][i]) + dR*(Q_modal[c][i+1]-Q_modal[c][i]) 
        end
    end
    # LF flux: left Boundary
    wavespd_curr = wavespeed_1D(Q_modal[1][1],Q_modal[2][1],Q_modal[3][1])
    wavespd_R = wavespeed_1D(Q_modal[1][2],Q_modal[2][2],Q_modal[3][2])
    wavespd_L = wavespeed_1D(rhoL,0.0,pL/(γ-1)) 
    dL = 1/2*max(wavespd_L,wavespd_curr)
    dR = 1/2*max(wavespd_R,wavespd_curr)
    visc[1][1] = dL*(rhoL-Q_modal[1][1]) + dR*(Q_modal[1][2]-Q_modal[1][1])
    visc[2][1] = dL*(0.0-Q_modal[2][1]) + dR*(Q_modal[2][2]-Q_modal[2][1])
    visc[3][1] = dL*(pL/(γ-1)-Q_modal[3][1]) + dR*(Q_modal[3][2]-Q_modal[3][1])
    # LF flux: right Boundary
    wavespd_curr = wavespeed_1D(Q_modal[1][end],Q_modal[2][end],Q_modal[3][end])
    wavespd_R = wavespeed_1D(rhoR,0.0,pR/(γ-1))
    wavespd_L = wavespeed_1D(Q_modal[1][end-1],Q_modal[2][end-1],Q_modal[3][end-1])
    dL = 1/2*max(wavespd_L,wavespd_curr)
    dR = 1/2*max(wavespd_R,wavespd_curr)
    visc[1][end] = dL*(Q_modal[1][end-1]-Q_modal[1][end]) + dR*(rhoR-Q_modal[1][end])
    visc[2][end] = dL*(Q_modal[2][end-1]-Q_modal[2][end]) + dR*(0.0-Q_modal[2][end])
    visc[3][end] = dL*(Q_modal[3][end-1]-Q_modal[3][end]) + dR*(pR/(γ-1)-Q_modal[3][end])


    #=
    for k = 1:K
        # LF flux
        #=
        for c = 1:3
            visc[c][1,k] = Q_modal[c][2,k]-Q_modal[c][1,k]
            visc[c][end,k] = Q_modal[c][end-1,k]-Q_modal[c][end,k]
            # visc[c][1,k] = Q_modal[c][2,k]-2*Q_modal[c][1,k]+Q_modal[c][end,mod1(k-1,K)]
            # visc[c][end,k] = Q_modal[c][end-1,k]-2*Q_modal[c][end,k]+Q_modal[c][1,mod1(k+1,K)]
            for i = 2:size(rhsQ[1],1)-1
                visc[c][i,k] = Q_modal[c][i-1,k]-2*Q_modal[c][i,k]+Q_modal[c][i+1,k]
            end
        end
        =#

        # # Roe flux (Matrix dissp) Roe_flux(rhoL,rhouL,EL,rhoR,rhouR,ER)
        # tmp1 = zeros(3)
        # tmp2 = zeros(3)
        #
        # tmp1 = Roe_flux(Q_modal[1][1,k],Q_modal[2][1,k],Q_modal[3][1,k],
        #                 Q_modal[1][2,k],Q_modal[2][2,k],Q_modal[3][2,k])
        # tmp2 = Roe_flux(Q_modal[1][1,k],Q_modal[2][1,k],Q_modal[3][1,k],
        #                 Q_modal[1][end,mod1(k-1,K)],Q_modal[2][end,mod1(k-1,K)],Q_modal[3][end,mod1(k-1,K)])
        # visc[1][1,k] = tmp1[1]+tmp2[1]
        # visc[2][1,k] = tmp1[2]+tmp2[2]
        # visc[3][1,k] = tmp1[3]+tmp2[3]
        # tmp1 = Roe_flux(Q_modal[1][end,k],Q_modal[2][end,k],Q_modal[3][end,k],
        #                 Q_modal[1][end-1,k],Q_modal[2][end-1,k],Q_modal[3][end-1,k])
        # tmp2 = Roe_flux(Q_modal[1][end,k],Q_modal[2][end,k],Q_modal[3][end,k],
        #                 Q_modal[1][1,mod1(k+1,K)],Q_modal[2][1,mod1(k+1,K)],Q_modal[3][1,mod1(k+1,K)])
        # visc[1][end,k] = tmp1[1]+tmp2[1]
        # visc[2][end,k] = tmp1[2]+tmp2[2]
        # visc[3][end,k] = tmp1[3]+tmp2[3]
        # for i = 2:size(rhsQ[1],1)-1
        #     tmp1 = Roe_flux(Q_modal[1][i,k],Q_modal[2][i,k],Q_modal[3][i,k],
        #                     Q_modal[1][i+1,k],Q_modal[2][i+1,k],Q_modal[3][i+1,k])
        #     tmp2 = Roe_flux(Q_modal[1][i,k],Q_modal[2][i,k],Q_modal[3][i,k],
        #                     Q_modal[1][i-1,k],Q_modal[2][i-1,k],Q_modal[3][i-1,k])
        #     visc[1][i,k] = tmp1[1]+tmp2[1]
        #     visc[2][i,k] = tmp1[2]+tmp2[2]
        #     visc[3][i,k] = tmp1[3]+tmp2[3]
        # end
    end
    #visc = (x->x.*repeat(is_shock',size(Q_modal[1],1))).(visc)
    =#

    #rhsQ = @. -(dfdx+rhsQ)
    rhsQ = @. -(dfdx+rhsQ-visc)
    rhsQ = (x->Vq*(x./J)).(rhsQ)
    return rhsQ
end

"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs_ES_nodal(Q,VDM,Vq,Vf,wq,wf,Pq,mapP,LIFT,nxJ,rxJ,Qrh_skew,J,K,compute_rhstest=false)
    # Assume Q is evaluation at nodal values
    Nq = size(Vq,1)
    Nf = size(Vf,1)
    Nh = Nq+Nf
    w = [wq;wf]

    Qq = (x->Vq*x).(Q)
    VU = v_ufun(Qq...)

    # TODO: when L2 project and interp at face quad, value blows up
    tmp = (x->Vf*Pq*x).(VU)
    Uf = u_vfun(tmp...)

    (rho,rhou,E) = vcat.(Q,Uf)

    beta = betafun(rho,rhou,E)
    Qh = (rho, rhou./rho, beta)
    QM = (x->x[Nq+1:Nh,:]).(Qh)
    QP = (x->x[mapP]).(QM)
    
    # Boundary condition
    QP[1][1] = rhoL
    QP[2][1] = 0.0
    QP[3][1] = betafun(rhoL,0.0,pL/(γ-1))
    QP[1][end] = rhoR
    QP[2][end] = 0.0
    QP[3][end] = betafun(rhoR,0.0,pR/(γ-1))
    
    (rhoM,rhouM,EM) = Uf
    rhoUM_n = @. rhouM*nxJ
    lam = abs.(wavespeed_1D(rhoM,rhoUM_n,EM))
    LFc = .5*max.(lam,lam[mapP])

    fSx = euler_fluxes(QM,QP)
    normal_flux(fx,u) = fx.*nxJ - LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,Uf)
    rhsQ = (x->LIFT*x).(flux)

    dfdx = [zeros(size(Qh[1])) for i in eachindex(Q)]
    for k = 1:K
        Qhk = [(x->x[i,k]).(Qh) for i = 1:Nh]
        Fk = [euler_fluxes(UL,UR) for UL in Qhk, UR in Qhk]
        for i = 1:3
            dfdx[i][:,k] += 2*(rxJ*Qrh_skew.*(x->x[i]).(Fk))*ones(Nh,1)
        end
    end

    dfdx = (x->[Pq LIFT]*diagm(1 ./ w)*x).(dfdx)

    # Viscosity
    VU_modal = (x->Pq*x).(VU)
    UV = u_vfun((x->Vq*x).(VU_modal)...)
    UV_modal = (x->Pq*x).(UV)

    coeff_1 = VDM\VU_modal[1]
    coeff_2 = VDM\VU_modal[1]
    coeff_3 = VDM\VU_modal[1]
    indicator_modal_V = zeros(K)
    for k = 1:K
        indicator_modal_V[k] += coeff_1[end,k]^2/sum(coeff_1[:,k].^2)
        indicator_modal_V[k] += coeff_2[end,k]^2/sum(coeff_2[:,k].^2)
        indicator_modal_V[k] += coeff_3[end,k]^2/sum(coeff_3[:,k].^2)
        indicator_modal_V[k] = indicator_modal_V[k]/3
    end
    is_shock = Float64.(indicator_modal_V .>= 1e-6)
    #is_shock = ones(length(indicator_modal_V))

    # indicator_proj_U = zeros(K)
    # for k = 1:K
    #     indicator_proj_U[k] += norm(UV[1][:,k]-Vq*UV_modal[1][:,k])^2
    #     indicator_proj_U[k] += norm(UV[2][:,k]-Vq*UV_modal[2][:,k])^2
    #     indicator_proj_U[k] += norm(UV[3][:,k]-Vq*UV_modal[3][:,k])^2
    #     indicator_proj_U[k] = sqrt(indicator_proj_U[k])
    # end
    # is_shock = Float64.(indicator_proj_U .>= 5e-3)

    visc = [zeros(size(rhsQ[1])), zeros(size(rhsQ[1])), zeros(size(rhsQ[1]))]
    Q_modal = (x->Pq*x).(Q)
    
    # LF flux
    for i = 2:K*size(rhsQ[1],1)-1
        wavespd_curr = wavespeed_1D(Q_modal[1][i],Q_modal[2][i],Q_modal[3][i])
        wavespd_R = wavespeed_1D(Q_modal[1][i+1],Q_modal[2][i+1],Q_modal[3][i+1])
        wavespd_L = wavespeed_1D(Q_modal[1][i-1],Q_modal[2][i-1],Q_modal[3][i-1])
        dL = 1/2*max(wavespd_L,wavespd_curr)
        dR = 1/2*max(wavespd_R,wavespd_curr)
        for c = 1:3
            visc[c][i] = dL*(Q_modal[c][i-1]-Q_modal[c][i]) + dR*(Q_modal[c][i+1]-Q_modal[c][i]) 
        end
    end
    # LF flux: left Boundary
    wavespd_curr = wavespeed_1D(Q_modal[1][1],Q_modal[2][1],Q_modal[3][1])
    wavespd_R = wavespeed_1D(Q_modal[1][2],Q_modal[2][2],Q_modal[3][2])
    wavespd_L = wavespeed_1D(rhoL,0.0,pL/(γ-1)) 
    dL = 1/2*max(wavespd_L,wavespd_curr)
    dR = 1/2*max(wavespd_R,wavespd_curr)
    visc[1][1] = dL*(rhoL-Q_modal[1][1]) + dR*(Q_modal[1][2]-Q_modal[1][1])
    visc[2][1] = dL*(0.0-Q_modal[2][1]) + dR*(Q_modal[2][2]-Q_modal[2][1])
    visc[3][1] = dL*(pL/(γ-1)-Q_modal[3][1]) + dR*(Q_modal[3][2]-Q_modal[3][1])
    # LF flux: right Boundary
    wavespd_curr = wavespeed_1D(Q_modal[1][end],Q_modal[2][end],Q_modal[3][end])
    wavespd_R = wavespeed_1D(rhoR,0.0,pR/(γ-1))
    wavespd_L = wavespeed_1D(Q_modal[1][end-1],Q_modal[2][end-1],Q_modal[3][end-1])
    dL = 1/2*max(wavespd_L,wavespd_curr)
    dR = 1/2*max(wavespd_R,wavespd_curr)
    visc[1][end] = dL*(Q_modal[1][end-1]-Q_modal[1][end]) + dR*(rhoR-Q_modal[1][end])
    visc[2][end] = dL*(Q_modal[2][end-1]-Q_modal[2][end]) + dR*(0.0-Q_modal[2][end])
    visc[3][end] = dL*(Q_modal[3][end-1]-Q_modal[3][end]) + dR*(pR/(γ-1)-Q_modal[3][end])


    #=
    for k = 1:K
        # LF flux
        #=
        for c = 1:3
            visc[c][1,k] = Q_modal[c][2,k]-Q_modal[c][1,k]
            visc[c][end,k] = Q_modal[c][end-1,k]-Q_modal[c][end,k]
            # visc[c][1,k] = Q_modal[c][2,k]-2*Q_modal[c][1,k]+Q_modal[c][end,mod1(k-1,K)]
            # visc[c][end,k] = Q_modal[c][end-1,k]-2*Q_modal[c][end,k]+Q_modal[c][1,mod1(k+1,K)]
            for i = 2:size(rhsQ[1],1)-1
                visc[c][i,k] = Q_modal[c][i-1,k]-2*Q_modal[c][i,k]+Q_modal[c][i+1,k]
            end
        end
        =#

        # # Roe flux (Matrix dissp) Roe_flux(rhoL,rhouL,EL,rhoR,rhouR,ER)
        # tmp1 = zeros(3)
        # tmp2 = zeros(3)
        #
        # tmp1 = Roe_flux(Q_modal[1][1,k],Q_modal[2][1,k],Q_modal[3][1,k],
        #                 Q_modal[1][2,k],Q_modal[2][2,k],Q_modal[3][2,k])
        # tmp2 = Roe_flux(Q_modal[1][1,k],Q_modal[2][1,k],Q_modal[3][1,k],
        #                 Q_modal[1][end,mod1(k-1,K)],Q_modal[2][end,mod1(k-1,K)],Q_modal[3][end,mod1(k-1,K)])
        # visc[1][1,k] = tmp1[1]+tmp2[1]
        # visc[2][1,k] = tmp1[2]+tmp2[2]
        # visc[3][1,k] = tmp1[3]+tmp2[3]
        # tmp1 = Roe_flux(Q_modal[1][end,k],Q_modal[2][end,k],Q_modal[3][end,k],
        #                 Q_modal[1][end-1,k],Q_modal[2][end-1,k],Q_modal[3][end-1,k])
        # tmp2 = Roe_flux(Q_modal[1][end,k],Q_modal[2][end,k],Q_modal[3][end,k],
        #                 Q_modal[1][1,mod1(k+1,K)],Q_modal[2][1,mod1(k+1,K)],Q_modal[3][1,mod1(k+1,K)])
        # visc[1][end,k] = tmp1[1]+tmp2[1]
        # visc[2][end,k] = tmp1[2]+tmp2[2]
        # visc[3][end,k] = tmp1[3]+tmp2[3]
        # for i = 2:size(rhsQ[1],1)-1
        #     tmp1 = Roe_flux(Q_modal[1][i,k],Q_modal[2][i,k],Q_modal[3][i,k],
        #                     Q_modal[1][i+1,k],Q_modal[2][i+1,k],Q_modal[3][i+1,k])
        #     tmp2 = Roe_flux(Q_modal[1][i,k],Q_modal[2][i,k],Q_modal[3][i,k],
        #                     Q_modal[1][i-1,k],Q_modal[2][i-1,k],Q_modal[3][i-1,k])
        #     visc[1][i,k] = tmp1[1]+tmp2[1]
        #     visc[2][i,k] = tmp1[2]+tmp2[2]
        #     visc[3][i,k] = tmp1[3]+tmp2[3]
        # end
    end
    #visc = (x->x.*repeat(is_shock',size(Q_modal[1],1))).(visc)
    =#

    #rhsQ = @. -(dfdx+rhsQ)
    rhsQ = @. -(dfdx+rhsQ-visc)
    rhsQ = (x->Vq*(x./J)).(rhsQ)
    return rhsQ
end






"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,10))/VDM
gr(size=(300,300),ylims=(0,1.2),legend=false,markerstrokewidth=1,markersize=2)
plt = plot(Vp*x,Vp*Pq*Q[1])


# filter_weights = ones(N+1)
# # filter_weights[end-3] = .5
# filter_weights[end-2] = .2
# filter_weights[end-1] = .1
# filter_weights[end] = .0
# Filter = VDM*(diagm(filter_weights)/VDM)

Q = collect(Q)
resQ = [zeros(size(xq)),zeros(size(xq)),zeros(size(xq))]
anim = @gif for i = 1:Nsteps
    for INTRK = 1:5
        #rhsQ = rhs_nodal(Q,Dr,LIFT,Vf,Pq,rxJ,J,nxJ,mapP)
        #rhsQ .= (x->Vq*Filter*Pq*x).(rhsQ)
        rhsQ = rhs_ES(Q,VDM,Vq,Vf,wq,wf,Pq,mapP,LIFT,nxJ,rxJ,Qrh_skew,J,K)

        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q   += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        # plot(Vp*x,Vp*Pq*Q[1],ylims=(0.0,3.0),title="Timestep $i out of $Nsteps",lw=2)
        # scatter!(x,Pq*Q[1])
        # sleep(.0)

        plot(Vp*x,Vp*Pq*Q[1],lw=2)
        #scatter!(x,Pq*Q[1])

        # VU = v_ufun(Q...)
        # VU_modal = (x->Pq*x).(VU)
        # UV = u_vfun((x->Vq*x).(VU_modal)...)
        # UV_modal = (x->Pq*x).(UV)

        # coeff_1 = VDM\VU_modal[1]
        # coeff_2 = VDM\VU_modal[1]
        # coeff_3 = VDM\VU_modal[1]
        # indicator_modal_V = zeros(K)
        # for k = 1:K
        #     indicator_modal_V[k] += coeff_1[end,k]^2/sum(coeff_1[:,k].^2)
        #     indicator_modal_V[k] += coeff_2[end,k]^2/sum(coeff_2[:,k].^2)
        #     indicator_modal_V[k] += coeff_3[end,k]^2/sum(coeff_3[:,k].^2)
        #     indicator_modal_V[k] = indicator_modal_V[k]/3
        # end
        #plot!(-0.45:0.1:0.45, indicator_modal_V*1e5,st=:bar,alpha=0.3)

        # indicator_proj_U = zeros(K)
        # for k = 1:K
        #     indicator_proj_U[k] += norm(UV[1][:,k]-Vq*UV_modal[1][:,k])^2
        #     indicator_proj_U[k] += norm(UV[2][:,k]-Vq*UV_modal[2][:,k])^2
        #     indicator_proj_U[k] += norm(UV[3][:,k]-Vq*UV_modal[3][:,k])^2
        #     indicator_proj_U[k] = sqrt(indicator_proj_U[k])
        # end
        #is_shock = Float64.(indicator_proj_U .>= 5e-3)
        #plot!(-0.45:0.1:0.45, indicator_proj_U*10,st=:bar,alpha=0.3)
    end
end every 100

# plot(Vp*x,Vp*Pq*Q[1],lw=2)
#scatter!(x,Pq*Q[1])


# coeff = VDM\(Pq*Q[1])
# indicator_modal = zeros(K)
# for k = 1:K
#     #indicator[k] = norm(Q[1][:,k]-Pq*Q[1][:,k])
#     indicator_modal[k] = coeff[end,k]^2/sum(coeff[:,k].^2)
# end
#
# VU = v_ufun(Q...)
# VU_modal = (x->Pq*x).(VU)
# UV = u_vfun((x->Vq*x).(VU_modal)...)
# UV_modal = (x->Pq*x).(UV)
#
# coeff_1 = VDM\VU_modal[1]
# coeff_2 = VDM\VU_modal[2]
# coeff_3 = VDM\VU_modal[3]
#
# indicator_proj_V = zeros(K)
# indicator_proj_U = zeros(K)
# indicator_L2err_V = zeros(K)
# indicator_L2err_U = zeros(K)
# indicator_L1err_V = zeros(K)
# indicator_L1err_U = zeros(K)
# indicator_L1err_normalized_V = zeros(K)
# indicator_L1err_normalized_U = zeros(K)
# indicator_modal_V1 = zeros(K)
# indicator_modal_V2 = zeros(K)
# indicator_modal_V3 = zeros(K)
# for k = 1:K
#     indicator_proj_V[k] = norm(VU[1][:,k]-Vq*VU_modal[1][:,k])
#     indicator_proj_U[k] = norm(UV[1][:,k]-Vq*UV_modal[1][:,k])
#     indicator_L2err_V[k] = sqrt(sum(wJq[:,k].*(VU[1][:,k]-Vq*VU_modal[1][:,k]).^2))
#     indicator_L2err_U[k] = sqrt(sum(wJq[:,k].*(UV[1][:,k]-Vq*UV_modal[1][:,k]).^2))
#     indicator_L1err_V[k] = sum(wJq[:,k].*abs.(VU[1][:,k]-Vq*VU_modal[1][:,k]))
#     indicator_L1err_U[k] = sum(wJq[:,k].*abs.(UV[1][:,k]-Vq*UV_modal[1][:,k]))
#     indicator_L1err_normalized_V[k] = sum(wJq[:,k].*abs.(VU[1][:,k]-Vq*VU_modal[1][:,k]))/sum(wJq[:,k].*maximum(abs.(VU[1][:,k])))
#     indicator_L1err_normalized_U[k] = sum(wJq[:,k].*abs.(UV[1][:,k]-Vq*UV_modal[1][:,k]))/sum(wJq[:,k].*maximum(abs.(UV[1][:,k])))
#     indicator_modal_V1[k] = coeff_1[end,k]^2/sum(coeff_1[:,k].^2)
#     indicator_modal_V2[k] = coeff_2[end,k]^2/sum(coeff_2[:,k].^2)
#     indicator_modal_V3[k] = coeff_3[end,k]^2/sum(coeff_3[:,k].^2)
# end
#
#
# # using DelimitedFiles
# #
# # open("indicator_modal.txt","w") do io
# #     writedlm(io,indicator_modal)
# # end
# #
# # open("indicator_proj_V.txt","w") do io
# #     writedlm(io,indicator_proj_V)
# # end
# #
# # open("indicator_proj_U.txt","w") do io
# #     writedlm(io,indicator_proj_U)
# # end
# #
# # open("indicator_L2err_V.txt","w") do io
# #     writedlm(io,indicator_L2err_V)
# # end
# #
# # open("indicator_L2err_U.txt","w") do io
# #     writedlm(io,indicator_L2err_U)
# # end
# #
# #
# # open("indicator_L1err_V.txt","w") do io
# #     writedlm(io,indicator_L1err_V)
# # end
# #
# # open("indicator_L1err_U.txt","w") do io
# #     writedlm(io,indicator_L1err_U)
# # end
# #
# # open("indicator_L1err_normalized_V.txt","w") do io
# #     writedlm(io,indicator_L1err_normalized_V)
# # end
# #
# # open("indicator_L1err_normalized_U.txt","w") do io
# #     writedlm(io,indicator_L1err_normalized_U)
# # end
# #
# # open("indicator_modal_V1.txt","w") do io
# #     writedlm(io,indicator_modal_V1)
# # end
# #
# # open("indicator_modal_V2.txt","w") do io
# #     writedlm(io,indicator_modal_V2)
# # end
# #
# # open("indicator_modal_V3.txt","w") do io
# #     writedlm(io,indicator_modal_V3)
# # end
#
#
#
# plot(Vp*x,Vp*Pq*Q[1],lw=2)
# scatter!(x,Pq*Q[1])
