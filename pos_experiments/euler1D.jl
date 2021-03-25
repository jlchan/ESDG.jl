using Plots

using UnPack
using Setfield
using StaticArrays

using RecursiveArrayTools

using LinearAlgebra
using NodesAndModes
using StartUpDG
using EntropyStableEuler
using FluxDiffUtils

# using OrdinaryDiffEq

N = 2
K1D = 20
CFL = .1
T = 1.0
interval = 100

# rd = RefElemData(Line(),N;
#                  quad_rule_vol=gauss_lobatto_quad(0,0,N))
rd = RefElemData(Line(),N)
VX,EToV = uniform_mesh(Line(),K1D)
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

###########################################
#      build hybridized SBP operators
###########################################

@unpack M,Dr,Vq,Pq,Vf,wf,nrJ = rd
Qr = Pq'*M*Dr*Pq
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
        -Br*Ef  Br]
Vh = [Vq;Vf]
Ph = M\transpose(Vh)
VhP = Vh*Pq

# make skew symmetric versions of the operators"
Qrhskew = .5*(Qrh-transpose(Qrh))
SBP_ops = (Matrix(Qrhskew'),Ph,VhP)

# interpolate geofacs to both vol/surf nodes, pack back into md
@unpack Vq,Vf = rd
@unpack rxJ = md
rxJ = [Vq;Vf]*rxJ # interp to hybridized points
@set md.rstxyzJ = SMatrix{1,1}([rxJ])

#######################################################
#      initial conditions                             #
#######################################################

@unpack x,xq,J = md
wJq = diagm(rd.wq)*(Vq*J)

# sine wave
rho = @. 1 + .5*sin(2*pi*x)
# rho = @. 1.0 + exp(-25*x^2)
u = @. .2 + 0*x
p = @. 20. + 0*x

# # non-dimensionalization
# p_ref = maximum(p)
# ρ_ref = minimum(rho)
# a_ref = sqrt(Euler{1}().γ*p_ref/ρ_ref)
# ρ = ρ/ρ_ref
# u = u/a_ref
# p = p/p_ref
# T = T*a_ref
# CFL = CFL/a_ref


function f(U)
    ρ,ρu,E = U
    u = ρu./ρ
    p = pfun(Euler{1}(),U)
    fm = @. ρu*u + p
    fE = @. (E + p)*u
    return SVector{3}(ρu,fm,fE)
end
fC(uL,uR) = .5 .* (f(uL) .+ f(uR))

# gaussian pulse
# rho = @. 1.0 + exp(-25*x^2)
# u = @. .0 + 0*ρ
# p = @. rho^Euler{1}().γ

# function sod(x)
#     ρ(x) = (x < 0) ? 1.0 : .125
#     u(x) = 0.0
#     p(x) = (x < 0) ? 1.0 : .1
#     return ρ.(x),u.(x),p.(x)
# end
# rho,u,p = (x->Pq*x).(sod(xq))

Q = VectorOfArray(prim_to_cons(Euler{1}(),SVector{3}(rho,u,p)))

#######################################################
#               local rhs computations                #
#######################################################

function rhs(Q, SBP_ops, rd::RefElemData, md::MeshData)

    QrTr,Ph,VhP = SBP_ops
    Ph *= 2.0
    @unpack LIFT,Vf,Vq = rd
    @unpack rxJ,J,K = md
    @unpack nxJ,sJ = md
    @unpack mapP,mapB = md

    Nh,Nq = size(VhP)

    # entropy var projection
    VU = (x->VhP*x).(v_ufun(Euler{1}(),(x->Vq*x).(Q.u)))
    Uh = u_vfun(Euler{1}(),VU)

    # compute face values
    Uf = (x->x[Nq+1:Nh,:]).(Uh)
    UP = (x->x[mapP]).(Uf)
    # UBC_left = prim_to_cons(Euler{1}(),sod(-1))
    # UBC_right = prim_to_cons(Euler{1}(),sod(1))
    # for fld = 1:length(UP)
    #     UP[fld][1] = UBC_left[fld]
    #     UP[fld][end] = UBC_right[fld]
    # end

    # simple lax friedrichs dissipation
    (rhoM,rhouM,EM) = Uf
    lam = abs.(rhouM./rhoM) + sqrt.(Euler{1}().γ*pfun(Euler{1}(),Uf)./rhoM)
    LFc = .25*max.(lam,lam[mapP]).*sJ

    fSx = fS(Euler{1}(),Uf,UP)
    # fSx = fC(Uf,UP)
    normal_flux(fx,u) = @. fx*nxJ - LFc*(u[mapP]-u)
    flux = map(normal_flux,fSx,Uf)
    rhsQ = (x->LIFT*x).(flux)

    # compute volume contributions using flux differencing
    rhse = zero.(getindex.(Uh,:,1))
    for e = 1:K
        fill!.(rhse,0.0)
        hadamard_sum_ATr!(rhse, (QrTr,), (x,y)->tuple(fS(Euler{1}(),x,y)),
                          getindex.(Uh,:,e); skip_index=(i,j)->(i>Nq)&&(j>Nq))
        # hadamard_sum_ATr!(rhse, (QrTr,), (x,y)->tuple(fC(x,y)),
        #                   getindex.(Uh,:,e); skip_index=(i,j)->(i>Nq)&&(j>Nq))
        for fld = 1:length(rhse)
            rhsQ[fld][:,e] .+= Ph*rhse[fld]
        end
    end

    return VectorOfArray(map(x -> -x./J,rhsQ))
end

CN = (N+1)^2/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)

# "Time integration coefficients"
rk4a,rk4b,rk4c = ck45()
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps
resQ = zero(Q)
unorm = zeros(Nsteps)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,SBP_ops,rd,md)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q   += rk4b[INTRK]*resQ
    end

    unorm[i] = sum(wJq.*S((x->Vq*x).(Q.u)))

    if i%interval==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
    end
end

# prob = ODEProblem(rhsQ!(rhs,Q.u,SBP_ops,rd,md).u,Q,(0.0,T))
# sol = solve(prob,Tsit5())
# Q = sol.u[end]

p1 = plot((x->rd.Vp*x).((x,Q[1])),lw=2,leg=false,lcolor=:black)

ΔS = unorm[end]-unorm[1]
s = "ΔS = " * sprintf1("%1.1e",ΔS)
p2 = plot((1:Nsteps)*dt,unorm,leg=false,xlims = (0,T),title=s)
plot(p1,p2)
# plot(p1)
