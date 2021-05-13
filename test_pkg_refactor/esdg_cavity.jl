using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using BenchmarkTools
using StaticArrays
using UnPack
using DelimitedFiles

using EntropyStableEuler.Fluxes2D
import EntropyStableEuler: γ

using StartUpDG
using StartUpDG.ExplicitTimestepUtils

using NodesAndModes

using FluxDiffUtils

"Approximation parameters"
N = 3 # The order of approximation
K1D = 4
CFL = 0.1
T = .10 # endtime
BCTYPE = 2 # 1 - adiabatic, 2 - Isothermal, 3 - Slip
TESTCASE = 1 # 1 - lid driven caivty, 2 - wave diffusion, 3 - shocktube
inviscid_dissp = true
viscous_dissp = false

"Viscous parameters"
Ma = .3
Re = 1000
mu = 1/Re
lambda = -2/3*mu
Pr = .71

# Time integration
CN = (N+1)*(N+2)/2  # estimated trace constant for CFL
h  = 2/K1D
dt = CFL * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps
dt0 = dt

"Mesh related variables"
VX, VY, EToV = uniform_tri_mesh(K1D, K1D)
# @. VX = 2*VX
# @. VY = VY + .25*sin(pi*VY)

# plot()
# fv = tri_face_vertices()
# for e = 1:size(EToV,1)
#     for f = 1:3
#         ids = EToV[e,fv[f]]
#         plot!(VX[ids],VY[ids],color=:black,lw=2,leg=false,border=:none)
#     end
# end
# display(plot!())
# png("convergence_mesh2.png")
# # display(scatter!(VX,VY,aspect_ratio=1))
# error("d")

# initialize ref element and mesh
rd = init_reference_tri(N)
md = init_DG_mesh(VX,VY,EToV,rd)

# LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
# build_periodic_boundary_maps!(md,rd,LX,zero.(LY))

# construct hybridized SBP operators
@unpack M,Dr,Ds,Vq,Pq,Vf,wf,nrJ,nsJ = rd
Qr = Pq'*M*Dr*Pq
Qs = Pq'*M*Ds*Pq
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Bs = diagm(wf.*nsJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
         -Br*Ef  Br]
Qsh = .5*[Qs-Qs' Ef'*Bs;
         -Bs*Ef  Bs]

Vh = [Vq;Vf]
Ph = M\transpose(Vh)
VhP = Vh*Pq

# make sparse skew symmetric versions of the operators"
# precompute union of sparse ids for Qr, Qs
Qrhskew = .5*(Qrh-transpose(Qrh))
Qshskew = .5*(Qsh-transpose(Qsh))

# interpolate geofacs to both vol/surf nodes
@unpack rxJ,sxJ,ryJ,syJ = md
rxJ,sxJ,ryJ,syJ = (x->Vh*x).((rxJ,sxJ,ryJ,syJ)) # interp to hybridized points
@pack! md = rxJ,sxJ,ryJ,syJ

# pack SBP operators into tuple
@unpack LIFT = rd
ops = (Qrhskew,Qshskew,VhP,Ph,LIFT,Vq)

function euler_fluxes_2D(UL,UR)
    rhoL = first(UL); betaL = last(UL)
    rhoR = first(UR); betaR = last(UR)
    logL = (log.(rhoL),log.(betaL))
    logR = (log.(rhoR),log.(betaR))
    return euler_fluxes(UL...,logL...,UR...,logR...)
end


function euler_fluxes_2D(UL,UR,logL,logR)
    return euler_fluxes(UL...,logL...,UR...,logR...)
end


function init_BC_funs(rd::RefElemData,md::MeshData,BCTYPE::Int64,Ma,γ)
    @unpack xf,yf,mapP,mapB,nxJ,nyJ,sJ = md
    xb,yb = (x->x[mapB]).((xf,yf))
    wsJ = diagm(rd.wf)*md.sJ

    # isothermal wall
    θwall = 1.0
    cv = 1 / (γ*(γ-1)*Ma^2) # WARNING: Ma = .3 - hack!

    # no-slip wall
    u1wall = 1.0
    u2wall = 0.0

    lid          = mapB[findall(@. abs(yb-1) < 1e-12)]
    wall         = mapB[findall(@. abs(yb-1) >= 1e-12)]
    bottomwall   = mapB[findall(@. abs(yb+1) < 1e-12)]
    leftwall     = mapB[findall(@. abs(xb+2) < 1e-12)]
    rightwall    = mapB[findall(@. abs(xb-2) < 1e-12)]
    boundary     = [lid;wall] # full boundary

    xlid = xf[lid]
    vlid = u1wall*ones(size(xlid))

    # # for periodic channel
    # wall = bottomwall
    # boundary = [lid;bottomwall]

    nxw = nxJ[wall]./sJ[wall]
    nyw = nyJ[wall]./sJ[wall]
    nxl = nxJ[lid]./sJ[lid]
    nyl = nyJ[lid]./sJ[lid]
    nx = nxJ[boundary]./sJ[boundary]
    ny = nyJ[boundary]./sJ[boundary]

    function impose_BCs_inviscid!(QP,Qf,md::MeshData)
        # No-slip at walls
        u_1 = Qf[2][boundary]
        u_2 = Qf[3][boundary]
        n_1 = nx
        n_2 = ny

        Un = @. u_1*n_1 + u_2*n_2
        Ut = @. u_1*n_2 - u_2*n_1

        # ρ^+ = ρ, p^+ = p (beta^+ = beta)
        @. QP[1][boundary] = Qf[1][boundary]
        @. QP[4][boundary] = Qf[4][boundary]

        # # u_n^+ = -u_n, u_t^+ = u_t
        # @. QP[2][boundary] = 1/(-n_1^2-n_2^2)*(n_1*Un-n_2*Ut)
        # @. QP[3][boundary] = 1/(-n_1^2-n_2^2)*(n_2*Un+n_1*Ut)
        @. QP[2][boundary] = Qf[2][boundary] - 2*Un*nx
        @. QP[3][boundary] = Qf[3][boundary] - 2*Un*ny
    end

    function impose_BCs_entropyvars!(VUP,VUf,md::MeshData)
        if BCTYPE == 1
            # Adiabatic no-slip BC
            @. VUP[2][wall] = -VUf[2][wall]
            @. VUP[3][wall] = -VUf[3][wall]
            @. VUP[4][wall] =  VUf[4][wall]

            @. VUP[2][lid] = -VUf[2][lid] - 2*vlid*VUf[4][lid]
            @. VUP[3][lid] = -VUf[3][lid]
            @. VUP[4][lid] =  VUf[4][lid]

        elseif BCTYPE == 2
            # # recompute θwall, θ
            # cvθ = 1.0/0.3^2/1.4/0.4

            # pwall = 1.0
            # cv = 1 / (γ*(γ-1)*Ma^2)
            # theta = @. pwall / (cv*(γ - 1))

            cvθ = cv*θwall # constant temperature along entire boundary

            # Isothermal BC
            @. VUP[2][wall] = -VUf[2][wall]
            @. VUP[3][wall] = -VUf[3][wall]
            @. VUP[4][wall] = -2.0/cvθ-VUf[4][wall]

            @. VUP[2][lid] = 2.0/cvθ-VUf[2][lid]
            @. VUP[3][lid] = -VUf[3][lid]
            @. VUP[4][lid] = -2.0/cvθ-VUf[4][lid]

        elseif BCTYPE == 3
            # Reflective BC
            v_1 = VUf[2][boundary]
            v_2 = VUf[3][boundary]
            n_1 = nx
            n_2 = ny

            VUn = @. v_1*n_1 + v_2*n_2
            VUt = @. v_1*n_2 - v_2*n_1

            # v_4^+ = v_4
            @. VUP[4][boundary] = VUf[4][boundary]
            # v_n^+ = -v_n, v_t^+ = v_t
            # @. VUP[2][boundary] = 1/(-n_1^2-n_2^2)*(n_1*VUn-n_2*VUt)
            # @. VUP[3][boundary] = 1/(-n_1^2-n_2^2)*(n_2*VUn+n_1*VUt)
            @. VUP[2][boundary] = VUf[2][boundary] - 2*VUn*nx
            @. VUP[3][boundary] = VUf[3][boundary] - 2*VUn*ny
        end
    end

    function impose_BCs_stress!(σxP,σyP,σxf,σyf,VUf,md::MeshData)

        visc_boundary_contribution = 0.0
        if BCTYPE == 1
            # Adiabatic no-slip BC
            @. σxP[2][wall] = σxf[2][wall]
            @. σyP[2][wall] = σyf[2][wall]
            @. σxP[3][wall] = σxf[3][wall]
            @. σyP[3][wall] = σyf[3][wall]

            @. σxP[2][lid] = σxf[2][lid]
            @. σyP[2][lid] = σyf[2][lid]
            @. σxP[3][lid] = σxf[3][lid]
            @. σyP[3][lid] = σyf[3][lid]

            @. σxP[4][wall] = -σxf[4][wall]
            @. σyP[4][wall] = -σyf[4][wall]
            @. σxP[4][lid]  = -σxf[4][lid] + 2*vlid*σxf[2][lid]
            @. σyP[4][lid]  = -σyf[4][lid] + 2*vlid*σyf[2][lid]
        elseif BCTYPE == 2
            # Isothermal
            @. σxP[2][boundary] = σxf[2][boundary]
            @. σyP[2][boundary] = σyf[2][boundary]
            @. σxP[3][boundary] = σxf[3][boundary]
            @. σyP[3][boundary] = σyf[3][boundary]
            @. σxP[4][boundary] = σxf[4][boundary]
            @. σyP[4][boundary] = σyf[4][boundary]

            qn = @. (-σxf[4][boundary]*nx - σyf[4][boundary]*ny)
            lid_ids = 1:length(lid)
            @. qn[lid_ids] += (u1wall*σxf[2][lid] + u2wall*σxf[3][lid])*nxl +
                (u1wall*σyf[2][lid] + u2wall*σyf[3][lid])*nyl   # u1 = 1, u2wall = 0, nx = 0, ny = 1 at lid
            visc_boundary_contribution = dot(wsJ[boundary],qn / (cv*θwall))

        elseif BCTYPE == 3
            sigma_x_1 = σxf[2][boundary]
            sigma_x_2 = σxf[3][boundary]
            sigma_y_1 = σyf[2][boundary]
            sigma_y_2 = σyf[3][boundary]
            n_1 = nx
            n_2 = ny

            σn_x = @. sigma_x_1*n_1 + sigma_x_2*n_2
            σn_y = @. sigma_y_1*n_1 + sigma_y_2*n_2

            @. σxP[2][boundary] = -σxf[2][boundary]+2*n_1*σn_x
            @. σyP[2][boundary] = -σyf[2][boundary]+2*n_1*σn_y
            @. σxP[3][boundary] = -σxf[3][boundary]+2*n_2*σn_x
            @. σyP[3][boundary] = -σyf[3][boundary]+2*n_2*σn_y

            @. σxP[4][boundary] = -σxf[4][boundary]
            @. σyP[4][boundary] = -σyf[4][boundary]
        end
        return visc_boundary_contribution
    end
    return impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress!
end
impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress! = init_BC_funs(rd,md,BCTYPE,Ma,γ)

"dense version - speed up by prealloc + transpose for col major "
function dense_hadamard_sum(Qhe,ops,vgeo,flux_fun)

    (Qr,Qs) = ops
    (rxJ,sxJ,ryJ,syJ) = vgeo
    # transpose for column-major evals
    QxTr = transpose(rxJ*Qr + sxJ*Qs)
    QyTr = transpose(ryJ*Qr + syJ*Qs)

    # precompute logs for logmean
    (rho,u,v,beta) = Qhe
    Qlog = (log.(rho), log.(beta))
    Qlogi = zeros(length(Qlog))
    Qlogj = zeros(length(Qlog))

    n = size(Qr,1)
    nfields = length(Qhe)

    QF = ntuple(x->zeros(n),nfields)
    QFi = zeros(nfields)
    for i = 1:n
        Qi = getindex.(Qhe,i)
        Qlogi = getindex.(Qlog,i)

        fill!(QFi,0)
        for j = 1:n
            Qj = getindex.(Qhe,j)
            Qlogj = getindex.(Qlog,j)

            Fx,Fy = flux_fun(Qi,Qj,Qlogi,Qlogj)
            @. QFi += QxTr[j,i]*Fx + QyTr[j,i]*Fy
        end

        for fld in eachindex(Qhe)
            QF[fld][i] = QFi[fld]
        end
    end

    return QF
end

function update_flux!(rhsQ,QM,QP,Uf,UP,LFc,nxJ,nyJ,mapP,Nfq,K,Nfields,inviscid_dissp)
    for k = 1:K
        for i = 1:Nfq
            fx,fy = euler_fluxes_2D((x->x[i,k]).(QP),(x->x[i,k]).(QM))
            tmp_nxJ = nxJ[i,k]
            tmp_nyJ = nyJ[i,k]
            tmp_LF = LFc[i,k]
            for d = 1:Nfields
                if inviscid_dissp
                    rhsQ[d][i,k] = fx[d]*tmp_nxJ+fy[d]*tmp_nyJ-tmp_LF*(UP[d][i,k]-Uf[d][i,k])
                else
                    rhsQ[d][i,k] = fx[d]*tmp_nxJ+fy[d]*tmp_nyJ
                end
            end
        end
    end
end

function flux_differencing!(QF,Qh,Qrh,Qsh,rxJ,ryJ,sxJ,syJ,Nh,Nq,Nfields,K)
    for k = 1:K
        rx = rxJ[1,k]
        ry = ryJ[1,k]
        sx = sxJ[1,k]
        sy = syJ[1,k]
        for j = 1:Nh # col idx
            for i = j:Nh # row idx
                if i <= Nq || j <= Nq # Skip lower right block
                    Fx,Fy = euler_fluxes_2D((x->x[i,k]).(Qh),(x->x[j,k]).(Qh))
                    Qr = Qrh[i,j]
                    Qs = Qsh[i,j]
                    for d = 1:Nfields
                        update_val = 2*((rx*Qr+sx*Qs)*Fx[d]+(ry*Qr+sy*Qs)*Fy[d])
                        QF[d][i,k] += update_val
                        QF[d][j,k] -= update_val
                    end
                end
            end
        end
    end
end

function rhs_inviscid!(Q,md::MeshData,ops,flux_fun,compute_rhstest,inviscid_dissp,
                        VU,Qh,QF,QM,QP,Uf,UP,rhsQ,tmp,tmp2,lam,LFc)
    @unpack rxJ,sxJ,ryJ,syJ,sJ,J,wJq = md
    @unpack nxJ,nyJ,sJ,mapP,mapB,K = md
    Qrh,Qsh,VhP,Ph,Lf,Vq = ops
    Nh,Nq = size(VhP)
    Nfq = size(Lf,2)
    Nfields = length(Q)

    fill!.(QF,0.0)

    # entropy var projection
    # VU = v_ufun((x->Vq*x).(Q)...)
    Q = (x->Vq*x).(Q)
    vector_norm(U) = sum((x->x.^2).(U))
    @. VU[3] = Q[2]^2+Q[3]^2 # rhoUnorm
    @. VU[4] = Q[4]-.5*VU[3]/Q[1] # rhoe
    @. VU[1] = log(0.4*VU[4]/(Q[1]^1.4)) # sU #TODO: hardcoded gamma
    @. VU[1] = (-Q[4]+VU[4]*(2.4-VU[1]))/VU[4]
    @. VU[2] = Q[2]/VU[4]
    @. VU[3] = Q[3]/VU[4]
    @. VU[4] = -Q[1]/VU[4]

    # VU = (x->VhP*x).(VU)
    Uh = (x->VhP*x).(VU)

    # Uh = u_vfun(VU...)
    @. tmp = Uh[2]^2+Uh[3]^2 #vUnorm
    @. tmp2 = (0.4/((-Uh[4])^1.4))^(1/0.4)*exp(-(1.4 - Uh[1] + tmp/(2*Uh[4]))/0.4) # rhoeV
    @. Uh[1] = tmp2.*(-Uh[4])
    @. Uh[2] = tmp2.*Uh[2]
    @. Uh[3] = tmp2.*Uh[3]
    @. Uh[4] = tmp2.*(1-tmp/(2*Uh[4]))

    # # convert to rho,u,v,beta vars
    # (rho,rhou,rhov,E) = Uh
    # beta = betafun(rho,rhou,rhov,E)
    # Qh = (rho, rhou./rho, rhov./rho, beta) # redefine Q = (rho,u,v,β)
    @. tmp = Uh[1]/(2*0.4*(Uh[4]-.5*(Uh[2]^2+Uh[3]^2)/Uh[1])) #beta
    @. Qh[1] = Uh[1]
    @. Qh[2] = Uh[2]./Uh[1]
    @. Qh[3] = Uh[3]./Uh[1]
    @. Qh[4] = tmp

    # # compute face values
    # QM = (x->x[Nq+1:Nh,:]).(Qh)
    # QP = (x->x[mapP]).(QM)
    # impose_BCs_inviscid!(QP,QM,md)
    for i = 1:Nfields
        QM[i] .= Qh[i][Nq+1:Nh,:]
        QP[i] .= QM[i][mapP]
    end
    impose_BCs_inviscid!(QP,QM,md)

    # simple lax friedrichs dissipation
    for i = 1:Nfields
        Uf[i] .= Uh[i][Nq+1:Nh,:]
    end
    (rhoM,rhouM,rhovM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    # lam = abs.(wavespeed(rhoM,rhoUM_n,EM))
    @. lam  = abs(sqrt(abs(rhoUM_n/rhoM))+sqrt(1.4*0.4*(EM-.5*rhoUM_n^2/rhoM)/rhoM))
    @. LFc = .25*max(lam,lam[mapP])*sJ

    for i = 1:Nfields
        UP[i] .= Uf[i][mapP]
    end
    update_flux!(rhsQ,QM,QP,Uf,UP,LFc,nxJ,nyJ,mapP,Nfq,K,Nfields,inviscid_dissp)
    rhsQ = (x->Lf*x).(rhsQ)

    flux_differencing!(QF,Qh,Qrh,Qsh,rxJ,ryJ,sxJ,syJ,Nh,Nq,Nfields,K)
    # hadamard_sum!(QF,(Qrh,Qsh),euler_fluxes,Qh)
    rhsQ = (x->Ph*x).(QF).+rhsQ
    rhsQ = (x -> -x./J).(rhsQ)

    rhstest = 0.0
    if compute_rhstest
        for fld in eachindex(rhsQ)
            VUq = VU[fld][1:Nq,:]
            rhstest += sum(wJq.*VUq.*(Vq*rhsQ[fld]))
        end
    end
    return rhsQ,rhstest
end

function dg_grad(Q,Qf,QP,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd

    Np = size(Dr,1)
    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qr = (x->Dr*x).(Q)
    Qs = (x->Ds*x).(Q)

    volx(ur,us) = @. rxj*ur + sxj*us
    voly(ur,us) = @. ryj*ur + syj*us
    surfx(uP,uf) = LIFT*(@. .5*(uP-uf)*nxJ)
    surfy(uP,uf) = LIFT*(@. .5*(uP-uf)*nyJ)
    rhsx = volx.(Qr,Qs) .+ surfx.(QP,Qf)
    rhsy = voly.(Qr,Qs) .+ surfy.(QP,Qf)
    return (x->x./J).(rhsx),(x->x./J).(rhsy)
end

function dg_grad!(Qx,Qy,Q,Qf,QP,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd
    Np = size(Dr,1)
    Nfields = size(Qx,1)

    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qr = (x->Dr*x).(Q)
    Qs = (x->Ds*x).(Q)

    volx(ur,us) = @. rxj*ur + sxj*us
    voly(ur,us) = @. ryj*ur + syj*us
    surfx(uP,uf) = LIFT*(@. .5*(uP-uf)*nxJ)
    surfy(uP,uf) = LIFT*(@. .5*(uP-uf)*nyJ)
    rhsx = volx.(Qr,Qs) .+ surfx.(QP,Qf)
    rhsy = voly.(Qr,Qs) .+ surfy.(QP,Qf)
    for i = 1:Nfields
        Qx[i] .= rhsx[i]./J
        Qy[i] .= rhsy[i]./J
    end
    #return (x->x./J).(rhsx),(x->x./J).(rhsy)
end

function dg_div(Qx,Qxf,QxP,Qy,Qyf,QyP,md::MeshData,rd::RefElemData)

    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd

    Np = size(Dr,1)
    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qxr = (x->Dr*x).(Qx)
    Qxs = (x->Ds*x).(Qx)
    Qyr = (x->Dr*x).(Qy)
    Qys = (x->Ds*x).(Qy)

    vol(uxr,uxs,uyr,uys) = @. rxj*uxr + sxj*uxs + ryj*uyr + syj*uys
    surf(uxP,uxf,uyP,uyf) = LIFT*(@. .5*((uxP-uxf)*nxJ + (uyP-uyf)*nyJ))
    rhs = vol.(Qxr,Qxs,Qyr,Qys) .+ surf.(QxP,Qxf,QyP,Qyf)

    return (x->x./J).(rhs)
end

function dg_div!(rhs,Qx,Qxf,QxP,Qy,Qyf,QyP,md::MeshData,rd::RefElemData)

    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,J = md
    @unpack Dr,Ds,LIFT,Vf = rd
    Nfields = length(Qx)

    Np = size(Dr,1)
    rxj,sxj,ryj,syj = (x->@view x[1:Np,:]).((rxJ,sxJ,ryJ,syJ))
    Qxr = (x->Dr*x).(Qx)
    Qxs = (x->Ds*x).(Qx)
    Qyr = (x->Dr*x).(Qy)
    Qys = (x->Ds*x).(Qy)

    vol(uxr,uxs,uyr,uys) = @. rxj*uxr + sxj*uxs + ryj*uyr + syj*uys
    surf(uxP,uxf,uyP,uyf) = LIFT*(@. .5*((uxP-uxf)*nxJ + (uyP-uyf)*nyJ))
    #rhs = vol.(Qxr,Qxs,Qyr,Qys) .+ surf.(QxP,Qxf,QyP,Qyf)
    for i = 1:Nfields
        rhs[i] .= (vol(Qxr[i],Qxs[i],Qyr[i],Qys[i]) .+ surf(QxP[i],Qxf[i],QyP[i],Qyf[i]))./J
    end
    #rhs .= (x->x./J).(rhs)
    #return (x->x./J).(rhs)
end

function init_visc_fxn(λ,μ,Pr)
    let λ=λ,μ=μ,Pr=Pr
        function viscous_matrices!(Kxx,Kxy,Kyy,v)
            v1,v2,v3,v4 = v
            inv_v4_cubed = @. 1/(v4^3)
            λ2μ = (λ+2.0*μ)
            Kxx[2,2] = inv_v4_cubed*-λ2μ*v4^2
            Kxx[2,4] = inv_v4_cubed*λ2μ*v2*v4
            Kxx[3,3] = inv_v4_cubed*-μ*v4^2
            Kxx[3,4] = inv_v4_cubed*μ*v3*v4
            Kxx[4,2] = inv_v4_cubed*λ2μ*v2*v4
            Kxx[4,3] = inv_v4_cubed*μ*v3*v4
            Kxx[4,4] = inv_v4_cubed*-(λ2μ*v2^2 + μ*v3^2 - γ*μ*v4/Pr)

            Kxy[2,3] = inv_v4_cubed*-λ*v4^2
            Kxy[2,4] = inv_v4_cubed*λ*v3*v4
            Kxy[3,2] = inv_v4_cubed*-μ*v4^2
            Kxy[3,4] = inv_v4_cubed*μ*v2*v4
            Kxy[4,2] = inv_v4_cubed*μ*v3*v4
            Kxy[4,3] = inv_v4_cubed*λ*v2*v4
            Kxy[4,4] = inv_v4_cubed*(λ+μ)*(-v2*v3)

            Kyy[2,2] = inv_v4_cubed*-μ*v4^2
            Kyy[2,4] = inv_v4_cubed*μ*v2*v4
            Kyy[3,3] = inv_v4_cubed*-λ2μ*v4^2
            Kyy[3,4] = inv_v4_cubed*λ2μ*v3*v4
            Kyy[4,2] = inv_v4_cubed*μ*v2*v4
            Kyy[4,3] = inv_v4_cubed*λ2μ*v3*v4
            Kyy[4,4] = inv_v4_cubed*-(λ2μ*v3^2 + μ*v2^2 - γ*μ*v4/Pr)
        end
        return viscous_matrices!
    end
end
viscous_matrices! = init_visc_fxn(lambda,mu,Pr)

function rhs_viscous!(Q,md::MeshData,rd::RefElemData,Re,BCTYPE,viscous_dissp,rhs,VU,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy)
    @unpack Pq,Vq,Vf,LIFT = rd
    @unpack K,mapP,mapB,J,wJq = md
    Nfields = length(Q)
    Nq = size(Vq,1)
    Np = size(Pq,1)
    Nfq = size(Vf,1)

    fill!.(rhs,0.0)
    fill!.(sigma_x,0.0)
    fill!.(sigma_y,0.0)

    # entropy var projection
    # VU = v_ufun((x->Vq*x).(Q)...)
    Q = (x->Vq*x).(Q)
    vector_norm(U) = sum((x->x.^2).(U))
    @. VU[3] = Q[2]^2+Q[3]^2 # rhoUnorm
    @. VU[4] = Q[4]-.5*VU[3]/Q[1] # rhoe
    @. VU[1] = log(0.4*VU[4]/(Q[1]^1.4)) # sU #TODO: hardcoded gamma
    @. VU[1] = (-Q[4]+VU[4]*(2.4-VU[1]))/VU[4]
    @. VU[2] = Q[2]/VU[4]
    @. VU[3] = Q[3]/VU[4]
    @. VU[4] = -Q[1]/VU[4]
    VU = (x->Pq*x).(VU)

    # compute and interpolate to quadrature
    VUf = (x->Vf*x).(VU)
    VUP = (x->x[mapP]).(VUf)
    impose_BCs_entropyvars!(VUP,VUf,md)

    dg_grad!(VUx,VUy,VU,VUf,VUP,md,rd)
    VUx = (x->Vq*x).(VUx)
    VUy = (x->Vq*x).(VUy)
    VUq = (x->Vq*x).(VU)

    # initialize sigma_x,sigma_y = viscous rhs
    vqi = zeros(Float64,Nfields)
    for e = 1:K
        # mult by matrices and perform local projections
        for i = 1:Nq
            @. vqi = getindex(VUq,i,e)
            viscous_matrices!(Kxx,Kxy,Kyy,vqi)
            for col = 2:Nfields
                vxi_col = VUx[col][i,e]
                vyi_col = VUy[col][i,e]
                for row = 2:Nfields
                    sigma_x[row][i,e] += Kxx[row,col]*vxi_col + Kxy[row,col]*vyi_col
                    sigma_y[row][i,e] += Kxy[col,row]*vxi_col + Kyy[row,col]*vyi_col
                end
            end
        end
    end

    # (Kij * Θj, θi)
    rhstest = 0.0
    for fld in 1:4
        rhstest += dot(wJq,@. VUx[fld]*sigma_x[fld] + VUy[fld]*sigma_y[fld])
    end

    sigma_x = (x->Pq*x).(sigma_x)
    sigma_y = (x->Pq*x).(sigma_y)

    sxf = (x->Vf*x).(sigma_x)
    syf = (x->Vf*x).(sigma_y)
    sxP = (x->x[mapP]).(sxf)
    syP = (x->x[mapP]).(syf)
    visc_boundary_contribution = impose_BCs_stress!(sxP,syP,sxf,syf,VUf,md)

    if viscous_dissp
        # add penalty
        #tau = .5
        tau = @. -1/Re/VUf[4]
        # tau = @. 10*(1/Re)*(-VUf[4])
        dV = ((xP,x)->xP-x).(VUP,VUf)
        avgV = ((xP,x)->1/2*(xP+x)).(VUP,VUf)
        @. penalization[2] = tau*dV[2]
        @. penalization[3] = tau*dV[3]
        @. penalization[4] = tau*dV[4]

        tau = tau[mapB]
        @. penalization[2][mapB] = tau*dV[2][mapB]
        @. penalization[3][mapB] = tau*dV[3][mapB]
        if BCTYPE == 1
            @. penalization[4][mapB] = -tau*(avgV[2][mapB]*dV[2][mapB]
                                            +avgV[3][mapB]*dV[3][mapB])/VUf[4][mapB]
        else
            @. penalization[4][mapB] = -tau*(avgV[2][mapB]*dV[2][mapB]
                                            +avgV[3][mapB]*dV[3][mapB]
                                            +  dV[4][mapB]*dV[4][mapB]/2)/VUf[4][mapB]
        end

        penalization = (x->LIFT*x).(penalization)
    end

    dg_div!(rhs,sigma_x,sxf,sxP,sigma_y,syf,syP,md,rd)

    if viscous_dissp
        return rhs .+ penalization, rhstest, visc_boundary_contribution
    else
        return rhs, rhstest, visc_boundary_contribution
    end
end




#####################
### Time Stepping ###
#####################
bcopy!(x,y) = x .= y

# define initial conditions at nodes
@unpack x,y = md
rho = ones(size(x))
u = zeros(size(x))
v = zeros(size(x))
p = (1/(Ma^2*γ))*ones(size(x))

if TESTCASE == 2
    # rho = @. 1.0 + exp(-25*(x^2+y^2))
    # u = zeros(size(x))
    # v = zeros(size(x))
    rho = one.(x)
    u = @. .1*sin(pi*x/2)*cos(pi/2*y)
    v = @. .1*cos(pi/2*x/2)*sin(pi*y)
    # p = (1/(Ma^2*γ))*ones(size(x))
    # p = @. rho^γ
elseif TESTCASE == 3
    rho_t(x) = x <= 0 ? 120.0 : 1.2
    rho = @. rho_t(x)
    u = zeros(size(x))
    v = zeros(size(x))
    p = @. rho/γ
end
# rho = ones(size(x))
# u = @. exp(-10*(((y-1)^2+x^2)))
# v = zeros(size(x))
# p = (1/(Ma^2*γ))*ones(size(x))

Q = primitive_to_conservative(rho,u,v,p)
Q = collect(Q) # make Q,resQ arrays of arrays for mutability
resQ = [zeros(size(x)) for i in eachindex(Q)]

@unpack Pq,Vq,Vf,LIFT = rd
@unpack K,mapP,mapB,J = md
Qrh,Qsh,VhP,Ph,Lf,Vq = ops
Nh,Nq = size(VhP)
Nfq,Np = size(Vf)
Nfields = length(Q)
make_zero_tuple(Nfields,Np,K) = ntuple(x->zeros(Np,K),Nfields)
rhs,VUx,VUy = ntuple(x->make_zero_tuple(Nfields,Np,K),3)
VU,sigma_x,sigma_y = ntuple(x->make_zero_tuple(Nfields,Nq,K),3)
QM,QP,Uf,UP,rhsQ_tmp,penalization = ntuple(x->make_zero_tuple(Nfields,Nfq,K),6)
Kxx,Kyy,Kxy = ntuple(x->MMatrix{4,4,Float64}(zeros(Nfields,Nfields)),3)
Qh,QF = ntuple(x->make_zero_tuple(Nfields,Nh,K),2)
tmp,tmp2 = ntuple(x->zeros(Float64,Nh,K),2)
lam,LFc = ntuple(x->zeros(Float64,Nfq,K),2)

function dopri45_coeffs()
    rk4a = [0.0             0.0             0.0             0.0             0.0             0.0         0.0
            0.2             0.0             0.0             0.0             0.0             0.0         0.0
            3.0/40.0        9.0/40.0        0.0             0.0             0.0             0.0         0.0
            44.0/45.0      -56.0/15.0       32.0/9.0        0.0             0.0             0.0         0.0
            19372.0/6561.0 -25360.0/2187.0  64448.0/6561.0  -212.0/729.0    0.0             0.0         0.0
            9017.0/3168.0  -355.0/33.0      46732.0/5247.0  49.0/176.0      -5103.0/18656.0 0.0         0.0
            35.0/384.0      0.0             500.0/1113.0    125.0/192.0     -2187.0/6784.0  11.0/84.0   0.0 ]

    rk4c = vec([0.0 0.2 0.3 0.8 8.0/9.0 1.0 1.0 ])

    # coefficients to evolve error estimator = b1-b2
    rk4E = vec([71.0/57600.0  0.0 -71.0/16695.0 71.0/1920.0 -17253.0/339200.0 22.0/525.0 -1.0/40.0 ])

    return rk4a,rk4E,rk4c
end

function rhsRK!(Q,rd,md,Re,BCTYPE,ops,euler_fluxes,inviscid_dissp,viscous_dissp,rhs,VU,Qh,QF,QM,QP,Uf,UP,rhsQ,tmp,tmp2,lam,LFc,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy)
    rhsQ,_ = rhs_inviscid!(Q,md,ops,euler_fluxes,false,inviscid_dissp,VU,Qh,QF,QM,QP,Uf,UP,rhsQ,tmp,tmp2,lam,LFc)
    visc_rhsQ,visc_test,visc_boundary_contribution = rhs_viscous!(Q,md,rd,Re,BCTYPE,viscous_dissp,rhs,VU,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy)
    bcopy!.(rhsQ, @. rhsQ + visc_rhsQ)

    let Pq=rd.Pq, Vq=rd.Vq, wJq=md.wJq
        rhstest = 0.0
        rhstest_visc = 0.0
        VU = v_ufun((x->Vq*x).(Q)...)
        VUq = (x->Vq*Pq*x).(VU)
        for field in eachindex(rhsQ)
            rhstest += sum(wJq.*VUq[field].*(Vq*rhsQ[field]))
            rhstest_visc += sum(wJq.*VUq[field].*(Vq*visc_rhsQ[field]))
        end
        rhstest_visc = rhstest_visc + visc_test
        return rhsQ,rhstest,rhstest_visc,visc_boundary_contribution
    end
end

rka,rkE,rkc = dopri45_coeffs()

# DOPRI storage
Qtmp = similar.(Q)
rhsQrk = ntuple(x->zero.(Q),length(rkE))

errEst = 0.0
prevErrEst = 0.0

t = 0.0
i = 0
interval = 5

dthist = Float64[dt]
thist = Float64[]
errhist = Float64[]
vischist = Float64[]
visc_boundary_hist = Float64[]
rhstesthist = Float64[]
wsJ = diagm(rd.wf)*md.sJ

prevQ = [zeros(size(x)) for i in eachindex(Q)]

rhsQ,_,_ = rhsRK!(Q,rd,md,Re,BCTYPE,ops,euler_fluxes_2D,inviscid_dissp,viscous_dissp,rhs,VU,Qh,QF,QM,QP,Uf,UP,rhsQ_tmp,tmp,tmp2,lam,LFc,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy)
bcopy!.(rhsQrk[1],rhsQ) # initialize DOPRI rhs (FSAL property)
@time begin
while t < T
    # DOPRI step and
    rhstest,rhstest_visc,visc_boundary_contrib = ntuple(x->0.0,3)
    for INTRK = 2:7
        k = zero.(Qtmp)
        for s = 1:INTRK-1
            bcopy!.(k, @. k + rka[INTRK,s]*rhsQrk[s])
        end
        bcopy!.(Qtmp, @. Q + dt*k)
        rhsQ,rhstest,rhstest_visc,visc_boundary_contrib = rhsRK!(Qtmp,rd,md,Re,BCTYPE,ops,euler_fluxes_2D,inviscid_dissp,viscous_dissp,rhs,VU,Qh,QF,QM,QP,Uf,UP,rhsQ_tmp,tmp,tmp2,lam,LFc,VUx,VUy,sigma_x,sigma_y,penalization,Kxx,Kyy,Kxy)
        bcopy!.(rhsQrk[INTRK],rhsQ)
    end
    errEstVec = zero.(Qtmp)
    for s = 1:7
        bcopy!.(errEstVec, @. errEstVec + rkE[s]*rhsQrk[s])
    end

    errTol = 5e-6
    errEst = 0.0
    for field = 1:length(Qtmp)
        errEstScale = @. abs(errEstVec[field]) / (errTol*(1+abs(Q[field])))
        errEst += sum(errEstScale.^2) # hairer seminorm
    end
    errEst = sqrt(errEst/(length(Q[1])*4))
    if errEst < 1.0 # if err small, accept step and update
            bcopy!.(Q, Qtmp)
            global t += dt
            bcopy!.(rhsQrk[1], rhsQrk[7]) # use FSAL property
    end
    order = 5
    dtnew = .8*dt*(.9/errEst)^(.4/(order+1)) # P controller
    if i > 0 # use PI controller if prevErrEst available
            dtnew *= (prevErrEst/max(1e-14,errEst))^(.3/(order+1))
    end
    global dt = max(min(10*dt0,dtnew),1e-9) # max/min dt
    global prevErrEst = errEst

    push!(dthist,dt)
    push!(thist,t)
    push!(vischist,rhstest_visc)
    push!(rhstesthist,rhstest)
    push!(visc_boundary_hist,visc_boundary_contrib)

    global i = i + 1  # number of total steps attempted
    if i%interval==0
        global prevQ
        preverr = sum(norm.(Q .- prevQ))
        bcopy!.(prevQ, Q)
        println("i = $i, t = $t, dt = $dtnew, errEst = $errEst, \nrhstest = $rhstest, rhstest_visc = $rhstest_visc, preverr = $preverr")
    end
end

end


################
### Plotting ###
################

#plotting nodes
@unpack Vp = rd
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)

xp = Vp*x
yp = Vp*y
Qp = (x->Vp*x).(Q)
vv = @. sqrt((Qp[2]^2 + Qp[3]^2)./Qp[1])
scatter(xp,yp,vv,zcolor=vv,camera=(0,90),colorbar=:right)

# @unpack xf,yf,mapP,mapB,nxJ,nyJ,sJ = md
# uf,vf = (x->(Vf*x)./(Vf*Q[1])).(Q[2:3])
# face_reshape(u) = reshape(u,size(u,1)÷3,3*md.K)
# Bfaces = findall(vec(md.FToF .== reshape(1:length(md.FToF),3,md.K)))
# get_bdry = (x->getindex(x,:,Bfaces))∘face_reshape
# xb,yb,ub,vb = get_bdry.((xf,yf,uf,vf))
#
# Qq = (x->Vq*x).(Q)
# VUf = (x->Vf*Pq*x).(v_ufun(Qq...))
# # Qf = (x->Vf*x).(Q)
# # VUf = v_ufun(Qf...)
# ub,vb = get_bdry.(VUf[2:3])
# wsJ = get_bdry(diagm(wf)*md.sJ)
#
# rq1D,w1D = gauss_quad(0,0,N)
# Vq1D = Line.vandermonde(N,rq1D)
# M1D = Vq1D'*diagm(w1D)*Vq1D
# Pq1D = M1D\(Vq1D'*diagm(w1D))
# rq1D_fine,w1D_fine =gauss_quad(0,0,N+2)
# xbq,ybq,ubq,vbq,wsJ = (x->Line.vandermonde(N,rq1D_fine) * Pq1D*x).((xb,yb,ub,vb,wsJ))
# @show sqrt(sum(@. wsJ*(ubq^2+vbq^2)))
#
# Vp1D = Line.vandermonde(N,LinRange(-1,1,100))
# VP1D = Vp1D*Pq1D
# xbp,ybp,ubp,vbp = (x->VP1D*x).((xb,yb,ub,vb))
#
# @show max(maximum(abs.(ubp)),maximum(abs.(vbp)))
#
# scatter(xbp,ybp,ubp,cam=(0,0),ms=1)
# scatter!(xbp,ybp,vbp,cam=(0,0))
