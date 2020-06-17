
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

"Constants"
const sp_tol = 1e-12

"Program parameters"
compute_L2_err = false

"Approximation Parameters"
N_P   = 2;    # The order of approximation in polynomial dimension
Np_P  = Int((N_P+1)*(N_P+2)/2)
Np_F  = 10;    # The order of approximation in Fourier dimension
K1D   = 30;   # Number of elements in polynomial (x,y) dimension
CFL   = 1.0;
T     = 0.5;  # End time

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
Dt = Array{Float64,2}(Toeplitz(column,column[[1;Np_F:-1:2]]))
D2t = Array{Float64,2}(Toeplitz(column2,column2[[1;Np_F:-1:2]]))
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
Ph = [Vq;Vf]*Pq # TODO: refactor
LIFTq = Vq*Lq

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

function flux_xy!(∇fh,Qh,Qlog,ops_flux,geo_flux,param_flux)
    K,Nq_P,Nh_P,Np_F,Nd = param_flux
    rxJ,sxJ,ryJ,syJ = geo_flux
    Qrh_skew,Qsh_skew,Qth,wq = ops_flux
    for k = 1:K
        for nf = 1:Np_F
            j = nf+Np_F*(k-1)
            update∇fh_xy!(∇fh,Qh,Qlog,Qrh_skew,Qsh_skew,Nh_P,Nd,j,rxJ[1,j],sxJ[1,j],ryJ[1,j],syJ[1,j])
        end
    end
end

#TODO: clean up function arguments
function update∇fh_xy!(∇fh,Qh,Qlog,Qrh_skew,Qsh_skew,Nh_P,Nd,j,rxJ,sxJ,ryJ,syJ)
    for col_idx = 1:Nh_P
        for row_idx = col_idx:Nh_P
            Fxj_tmp, Fyj_tmp,_ = euler_fluxes(Qh[1][row_idx,j],Qh[2][row_idx,j],Qh[3][row_idx,j],Qh[4][row_idx,j],Qh[5][row_idx,j],
                                               Qh[1][col_idx,j],Qh[2][col_idx,j],Qh[3][col_idx,j],Qh[4][col_idx,j],Qh[5][col_idx,j],
                                               Qlog[1][row_idx,j],Qlog[2][row_idx,j],
                                               Qlog[1][col_idx,j],Qlog[2][col_idx,j])
            var_Qrh = Qrh_skew[row_idx,col_idx]
            var_Qsh = Qsh_skew[row_idx,col_idx]
            for i = 1:Nd
                update_val = 2*((rxJ*var_Qrh+sxJ*var_Qsh)*Fxj_tmp[i]
                               +(ryJ*var_Qrh+syJ*var_Qsh)*Fyj_tmp[i])
                ∇fh[i][row_idx,j] += update_val
                ∇fh[i][col_idx,j] -= update_val
            end
        end
    end
end

function flux_z!(∇fh,Qh,Qlog,ops_flux,geo_flux,param_flux)
    K,Nq_P,Nh_P,Np_F,Nd = param_flux
    rxJ,sxJ,ryJ,syJ = geo_flux
    Qrh_skew,Qsh_skew,Qth,wq = ops_flux
    for k = 1:K
        j_idx = (k-1)*Np_F+1:k*Np_F
        for nh = 1:Nq_P
            update∇fh_z!(∇fh,Qth,Np_F,Nd,nh,wq,J,k,Qlog,Qh)
        end
    end
end

# TODO: clean up
function update∇fh_z!(∇fh,Qth,Np_F,Nd,nh,wq,J,k,Qlog,Qh)
    j_idx = (k-1)*Np_F+1:k*Np_F
    wqn = 2/J*wq[nh]
    for col_idx = 1:Np_F
        for row_idx = col_idx:Np_F
            _,_,f_tmp = euler_fluxes((x->x[nh,j_idx[row_idx]]).(Qh),(x->x[nh,j_idx[col_idx]]).(Qh),(x->x[nh,j_idx[row_idx]]).(Qlog),(x->x[nh,j_idx[col_idx]]).(Qlog))
            var_Qth = wqn*Qth[row_idx,col_idx]
            for i = 1:Nd
                ∇fh[i][nh,j_idx[row_idx]] += var_Qth*f_tmp[i]
                ∇fh[i][nh,j_idx[col_idx]] -= var_Qth*f_tmp[i]
            end
        end
    end
end

function update_rhs_flux!(rhsQ,Nh_P,Nq_P,K,Np_F,Nfp_P,mapP,Nd,Qh,nxJ,nyJ,Qlog)
    for col_idx = 1:K*Np_F
        for row_idx = 1:Nfp_P
            tmp_idx = mapP[row_idx+(col_idx-1)*Nfp_P]
            r_idx = Nq_P+Nh_P*div(tmp_idx-1,Nfp_P)+mod1(tmp_idx,Nfp_P)
            tmp_flux_x, tmp_flux_y,_ = euler_fluxes((x->x[Nq_P+row_idx,col_idx]).(Qh),(x->x[r_idx]).(Qh),
                                                    (x->x[Nq_P+row_idx,col_idx]).(Qlog),(x->x[r_idx]).(Qlog))
            tmp_nxJ = nxJ[row_idx,col_idx]
            tmp_nyJ = nyJ[row_idx,col_idx]
            for d = 1:Nd
                # normal_flux(fx,fy,u) = fx.*nxJ + fy.*nyJ - LFc.*(u[mapP]-u)
                # TODO: Lax Friedrichs
                rhsQ[d][row_idx,col_idx] = tmp_flux_x[d]*tmp_nxJ + tmp_flux_y[d]*tmp_nyJ
            end
        end
    end
end

# TODO: refactor
ops = (Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq)
mesh = (rxJ,sxJ,ryJ,syJ,sJ,nxJ,nyJ,JP,JF,J,h,mapM,mapP)
param = (K,Np_P,Nfp_P,Np_F,Nq_P,Nh_P)
function rhs(Q,ops,mesh,param,compute_rhstest)

    Vq,Vf,wq,wf,Pq,Lq,Qrh_skew,Qsh_skew,Qth,Ph,LIFTq = ops
    rxJ,sxJ,ryJ,syJ,sJ,nxJ,nyJ,JP,JF,J,h,mapM,mapP = mesh
    K,Np_P,Nfp_P,Np_F,Nq_P,Nh_P = param
    Nd = length(Q) # number of components

    rhsQ = (zeros(Nfp_P,K*Np_F),zeros(Nfp_P,K*Np_F),zeros(Nfp_P,K*Np_F),zeros(Nfp_P,K*Np_F),zeros(Nfp_P,K*Np_F))
    VU = (zeros(Nq_P,K*Np_F),zeros(Nq_P,K*Np_F),zeros(Nq_P,K*Np_F),zeros(Nq_P,K*Np_F),zeros(Nq_P,K*Np_F))
    Qh = (zeros(Nh_P,K*Np_F),zeros(Nh_P,K*Np_F),zeros(Nh_P,K*Np_F),zeros(Nh_P,K*Np_F),zeros(Nh_P,K*Np_F))
    tmp = zeros(Nh_P,K*Np_F)
    tmp2 = zeros(Nh_P,K*Np_F)
    vector_norm(U) = sum((x->x.^2).(U))
    # VU = v_ufun(Q...)
    @. VU[4] = Q[2]^2+Q[3]^2+Q[4]^2 # rhoUnorm
    @. VU[5] = Q[5]-.5*VU[4]/Q[1] # rhoe
    @. VU[1] = log(0.4*VU[5]/(Q[1]^1.4)) # sU #TODO: hardcoded gamma
    @. VU[1] = (-Q[5]+VU[5]*(2.4-VU[1]))/VU[5]
    @. VU[2] = Q[2]/VU[5]
    @. VU[3] = Q[3]/VU[5]
    @. VU[4] = Q[4]/VU[5]
    @. VU[5] = -Q[1]/VU[5]
    # @show maximum.(VU)
    # @show minimum.(VU)
    # @show sum.(VU)

    Qh = (x->Ph*x).(VU)
    # (ρ,ρu,ρv,ρw,E) = u_vfun(Qh...)
    @. tmp = Qh[2]^2+Qh[3]^2+Qh[4]^2 #vUnorm
    @. tmp2 = (0.4/((-Qh[5])^1.4))^(1/0.4)*exp(-(1.4 - Qh[1] + tmp/(2*Qh[5]))/0.4) # rhoeV
    @. Qh[1] = tmp2.*(-Qh[5])
    @. Qh[2] = tmp2.*Qh[2]
    @. Qh[3] = tmp2.*Qh[3]
    @. Qh[4] = tmp2.*Qh[4]
    @. Qh[5] = tmp2.*(1-tmp/(2*Qh[5]))
    # @show maximum.(Qh)
    # @show minimum.(Qh)
    # @show sum.(Qh)

    # TODO: Lax Friedrichs Dissipation flux
    # (ρM,ρuM,ρvM,ρwM,EM) = Uf
    # ρuM_n = @. (ρuM*nxJ+ρvM*nyJ)/sJ # TODO: 3D lax-friedrichs?
    # lam = abs.(wavespeed(ρM,ρuM_n,EM))
    # LFc = .5*max.(lam,lam[mapP]).*sJ

    # (Qh[2][Nq_P+1:end,:].*nxJ+Qh[3][Nq_P+1:end,:].*nyJ)./sJ

    # β = betafun(ρ,ρu,ρv,ρw,E)
    # Qh2 = (ρ,ρu./ρ,ρv./ρ,ρw./ρ,β)
    @. tmp = Qh[1]/(2*0.4*(Qh[5]-.5*(Qh[2]^2+Qh[3]^2+Qh[4]^2)/Qh[1])) #beta
    @. Qh[2] = Qh[2]./Qh[1]
    @. Qh[3] = Qh[3]./Qh[1]
    @. Qh[4] = Qh[4]./Qh[1]
    @. Qh[5] = tmp
    # @show maximum.(Qh)
    # @show minimum.(Qh)
    # @show sum.(Qh)

    # TODO: implement Lax Friedrichs
    # TODO: storing flux, so avoid calculate flux on face quad point again?
    Qlog = (log.(Qh[1]),log.(Qh[5]))
    update_rhs_flux!(rhsQ,Nh_P,Nq_P,K,Np_F,Nfp_P,mapP,Nd,Qh,nxJ,nyJ,Qlog)
    # @show maximum.(rhsQ)
    rhsQ = (x->Vq*Lq*x).(rhsQ) # TODO: put it into update_rhs_flux!
    # @show maximum.(Qlog)
    # @show maximum.(rhsQ)

    # Flux differencing
    # TODO: cleaner syntax, avoid storage?
    ∇fh = (zeros(size(Qh[1])),zeros(size(Qh[1])),zeros(size(Qh[1])),zeros(size(Qh[1])),zeros(size(Qh[1])))

    # @show "========= Flux Differencing =========="
    # TODO: fix Jacobian
    param_flux = (K,Nq_P,Nh_P,Np_F,Nd)
    geo_flux = (rxJ,sxJ,ryJ,syJ)
    ops_flux = (Qrh_skew,Qsh_skew,Qth,wq)
    flux_xy!(∇fh,Qh,Qlog,ops_flux,geo_flux,param_flux)
    # @show maximum.(∇fh)
    flux_z!(∇fh,Qh,Qlog,ops_flux,geo_flux,param_flux)
    # @show maximum.(∇fh)
    ∇f = (x->Vq*[Pq Lq]*diagm(1 ./ [wq;wf])*x).(∇fh)
    # @show maximum.(∇f)
    rhsQ = @. -(∇f+rhsQ)
    # @show maximum.(rhsQ)

    rhstest = 0
    if compute_rhstest
        for fld in eachindex(rhsQ)
            rhstest += sum(wq.*VU[fld].*rhsQ[fld])
        end
    end
    # rhstest = 0
    # rhsQ = [zeros(size(Q[1])), zeros(size(Q[1])),zeros(size(Q[1])),zeros(size(Q[1])),zeros(size(Q[1]))]
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
@time begin
for i = 1:Nsteps
    rhstest = 0
    for INTRK = 1:5
        # @show "===================="
        # @show INTRK
        compute_rhstest = INTRK==5
        rhsQ,rhstest = rhs(Q,ops,mesh,param,compute_rhstest)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")
    end
end

end # time

rq2,sq2,wq2 = quad_nodes_2D(N_P+2)
Vq2 = vandermonde_2D(N_P,rq2,sq2)/VDM
xq2,yq2,zq2 = (x->Vq2*x).((x,y,z))
ρ = Vq2*Pq*Q[1]
ρ_ex = ρ_exact(xq2,yq2,zq2,T)
Q = (x->Vq2*Pq*x).(Q)
p = pfun(Q[1],(Q[2],Q[3],Q[4]),Q[5])
Q = (Q[1],Q[2]./Q[1],Q[3]./Q[1],Q[4]./Q[1],p)
Q_ex = Q_exact(xq2,yq2,zq2,T)

L2_err = 0.0
for fld in eachindex(Q)
    global L2_err
    L2_err += sum(h*J*wq2.*(Q[fld]-Q_ex[fld]).^2)
end
println("L2err at final time T = $T is $L2_err\n")
