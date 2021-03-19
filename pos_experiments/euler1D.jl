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

N = 2
K1D = 400
CFL = .25
T = .2
interval = 50

rd = RefElemData(Line(),N;
                 quad_rule_vol=gauss_quad(0,0,N+1))
VX,EToV = uniform_mesh(Line(),K1D)
md = MeshData(VX,EToV,rd)
# md = make_periodic(md,rd)

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

@unpack x,xq = md
rho = @. 2 + .5*exp(-25*(x^2))
u = @. 0*x
p = @. rho^Euler{1}().γ

function sod(x)
    ρ(x) = (x < 0) ? 1.0 : .125
    u(x) = 0.0
    p(x) = (x < 0) ? 1.0 : .1
    return ρ.(x),u.(x),p.(x)
end

rho,u,p = (x->Pq*x).(sod(xq))

Q = VectorOfArray(prim_to_cons(Euler{1}(),SVector{3}(rho,u,p)))

#######################################################
#               local rhs computations                #
#######################################################

function init_mxm(Ph)
    mxm_accum!(X,x,e) = X[:,e] .+= 2.0*Ph*x
    return mxm_accum!
end
mxm_acc! = init_mxm(Ph)

function rhs(Q, SBP_ops, rd::RefElemData, md::MeshData)

    QrTr,Ph,VhP = SBP_ops
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
    UBC_left = prim_to_cons(Euler{1}(),sod(-1))
    UBC_right = prim_to_cons(Euler{1}(),sod(1))
    for fld = 1:length(UP)
        UP[fld][1] = UBC_left[fld]
        UP[fld][end] = UBC_right[fld]
    end

    # simple lax friedrichs dissipation
    (rhoM,rhouM,EM) = Uf
    lam = abs.(rhouM./rhoM) + sqrt.(Euler{1}().γ*pfun(Euler{1}(),Uf)./rhoM)
    LFc = .25*max.(lam,lam[mapP]).*sJ

    fSx = fS(Euler{1}(),Uf,UP)
    normal_flux(fx,u) = @. fx*nxJ - LFc*(u[mapP]-u)
    flux = map(normal_flux,fSx,Uf)
    rhsQ = (x->LIFT*x).(flux)

    # compute volume contributions using flux differencing
    rhse = zero.(getindex.(Uh,:,1))
    for e = 1:K
        fill!.(rhse,0.0)
        hadamard_sum_ATr!(rhse, (rxJ[1,e]*QrTr,),
                          (x,y)->tuple(fS(Euler{1}(),x,y)),
                          getindex.(Uh,:,e))
        for fld = 1:length(rhse)
            rhsQ[fld][:,e] .+= 2.0*Ph*rhse[fld]
        end

        # mxm_acc!.(rhsQ,rhse,e)
    end

    return VectorOfArray(map(x -> -x./J,rhsQ))
end

# "Time integration coefficients"
rk4a,rk4b,rk4c = ck45()
CN = (N+1)^2/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# "Perform time-stepping"
resQ = zero(Q)
#@gif
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,SBP_ops,rd,md)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q   += rk4b[INTRK]*resQ
    end

    if i%interval==0 || i==Nsteps
        println("Time step $i out of $Nsteps")
        # scatter((x->rd.Vp*x).((x,Q[1])),zcolor=rd.Vp*Q[1],msw=0,ms=2,cam=(0,90))
    end
end #every interval

scatter((x->rd.Vp*x).((x,Q[1])),zcolor=rd.Vp*Q[1],msw=0,ms=2)
