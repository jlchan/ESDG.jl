using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using StaticArrays # for MArrays
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./src")
using CommonUtils
using NodesAndModes
using NodesAndModes.Tri

using UniformTriMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

"Approximation parameters"
N = 1 # The order of approximation
K1D = 10
CFL = .15 # CFL goes up to 2.5ish
T = 2 #1.95 #1.0 # endtime

"Viscous parameters"
mu = 1e-4
lambda = -2/3*mu
Pr = .72

"Mesh related variables"
Kx = 2*K1D
Ky = K1D
VX, VY, EToV = uniform_tri_mesh(Kx, Ky)
@. VX = (1+VX)
@. VY = (1+VY)/2

# initialize ref element and mesh
# specify lobatto nodes to automatically get DG-SEM mass lumping
rd = init_reference_tri(N,Nq=2*N+1)
md = init_mesh((VX,VY),EToV,rd)

# # Make domain periodic
# @unpack Nfaces,Vf = rd
# @unpack xf,yf,K,mapM,mapP,mapB = md
# LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
# mapPB = build_periodic_boundary_maps(xf,yf,LX,0*LY,Nfaces*K,mapM,mapP,mapB)
# mapP[mapB] = mapPB
# @pack! md = mapP

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

# define initial conditions at nodes
@unpack x,y = md

function freestream_primvars() # ρ,u,v,p
    return 1,1,0,4/γ
end
U = ntuple(a->zeros(size(x)),4)
for fld = 1:4
    U[fld] .= freestream_primvars()[fld]
end
Q = primitive_to_conservative(U...)

# interpolate geofacs to both vol/surf nodes
@unpack rxJ, sxJ, ryJ, syJ = md
rxJ, sxJ, ryJ, syJ = (x->Vh*x).((rxJ, sxJ, ryJ, syJ)) # interp to hybridized points
@pack! md = rxJ, sxJ, ryJ, syJ

# pack SBP operators into tuple
@unpack LIFT = rd
ops = (Qrhskew,Qshskew,VhP,Ph,LIFT,Vq)

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)/2  # estimated trace constant for CFL
h  = 2/K1D
dt = CFL * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

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

    n = size(Qr,1)
    nfields = length(Qhe)

    QF = zero.(Qhe) #ntuple(x->zeros(n),nfields)
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

"Qh = (rho,u,v,beta), while Uh = conservative vars"
function rhs(Q,md::MeshData,ops,flux_fun::Fxn) where Fxn

    @unpack rxJ,sxJ,ryJ,syJ,J,wJq = md
    @unpack nxJ,nyJ,sJ,mapP,mapB,K = md
    Qrh,Qsh,VhP,Ph,Lf,Vq = ops
    Nh,Nq = size(VhP)

    # entropy var projection
    VU = v_ufun((x->Vq*x).(Q)...)
    VU = (x->VhP*x).(VU)
    Uh = u_vfun(VU...)

    # convert to rho,u,v,beta vars
    (rho,rhou,rhov,E) = Uh
    beta = betafun(rho,rhou,rhov,E)
    Qh = (rho, rhou./rho, rhov./rho, beta) # redefine Q = (rho,u,v,β)

    # compute face values
    QM = (x->x[Nq+1:Nh,:]).(Qh)
    QP = (x->x[mapP]).(QM)
    impose_BCs_Qvars!(QP,QM,md)

    # simple lax friedrichs dissipation
    Uf =  (x->x[Nq+1:Nh,:]).(Uh)
    (rhoM,rhouM,rhovM,EM) = Uf
    rhoUM_n = @. (rhouM*nxJ + rhovM*nyJ)/sJ
    lam = abs.(wavespeed(rhoM,rhoUM_n,EM))
    LFc = .25*max.(lam,lam[mapP]).*sJ

    fSx,fSy = flux_fun(QM,QP)
    normal_flux(fx,fy,u) = fx.*nxJ + fy.*nyJ - LFc.*(u[mapP]-u)
    flux = normal_flux.(fSx,fSy,Uf)
    rhsQ = (x->Lf*x).(flux)

    mxm_accum!(X,x,e) = X[:,e] .+= 2*Ph*x

    # compute volume contributions using flux differencing
    for e = 1:K
        Qhe = getindex.(Qh,:,e) # force tuples for fast splatting
        vgeo_local = getindex.((rxJ,sxJ,ryJ,syJ),1,e) # assumes affine elements for now

        Qops = (Qrh,Qsh)
        QFe = dense_hadamard_sum(Qhe,Qops,vgeo_local,flux_fun)

        mxm_accum!.(rhsQ,QFe,e)
    end

    rhsQ = (x -> -x./J).(rhsQ)
    return rhsQ # scale by Jacobian
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

function init_visc_fxn(λ,μ,Pr)
    let λ=λ,μ=μ,Pr=Pr
        function viscous_matrices!(Kxx,Kxy,Kyy,v)
            v1,v2,v3,v4 = v
            inv_v4_cubed = @. 1/(v4^3)
            λ2μ = (λ+2.0*μ)
            Kxx .= inv_v4_cubed*[0. 0. 0. 0.;
                        0. -λ2μ*v4^2 0. λ2μ*v2*v4;
                        0. 0. -μ*v4^2 μ*v3*v4;
                        0. λ2μ*v2*v4 μ*v3*v4 -(λ2μ*v2^2 + μ*v3^2 - γ*μ*v4/Pr)]
            Kxy .= inv_v4_cubed*[0. 0. 0. 0.;
                        0. 0. -λ*v4^2 λ*v3*v4;
                        0. -μ*v4^2 0. μ*v2*v4;
                        0. μ*v3*v4 λ*v2*v4 (λ+μ)*(-v2*v3)]
            Kyy .= inv_v4_cubed*[0. 0. 0. 0.;
                        0. -μ*v4^2 0. μ*v2*v4;
                        0. 0. -λ2μ*v4^2 λ2μ*v3*v4;
                        0. μ*v2*v4 λ2μ*v3*v4 -(λ2μ*v3^2 + μ*v2^2 - γ*μ*v4/Pr)]
        end
        return viscous_matrices!
    end
end
viscous_matrices! = init_visc_fxn(lambda,mu,Pr)

function init_BC_funs(md::MeshData)
    @unpack xf,yf,mapP,mapB,nxJ,nyJ = md
    xb,yb = (x->x[mapB]).((xf,yf))

    #  _____
    # |     |
    # |_____|
    # flat plate is x > 1 bottom part
    tol = 1e-13
    top    = @. abs(yb-1)<tol
    bottom_left_wall = @. (@. xb < 1) & (@. abs(yb)<tol)

    # define boundary regions
    inflow   = mapB[findall(@. abs(xb)<tol)]
    outflow  = mapB[findall(@. abs(xb-2)<tol)]
    wall     = mapB[findall((@. abs(yb)<tol) .& (@. xb > 1))]
    symmetry = mapB[findall(@. top | bottom_left_wall)]

    # nxb,nyb =

    function impose_BCs_Qvars!(QP,Qf,md::MeshData)
        ρ_∞,u_∞,v_∞,p_∞ = freestream_primvars()
        β_∞ = betafun(primitive_to_conservative(ρ_∞,u_∞,v_∞,p_∞)...)
        QP[1][inflow] .= ρ_∞
        QP[2][inflow] .= u_∞
        QP[3][inflow] .= v_∞
        QP[4][inflow] .= β_∞

        # impose mirror states (no-normal flow) at wall (has normal (0,1))
        QP[2][wall] .= Qf[2][wall]
        QP[3][wall] .= -Qf[3][wall] # yvelocity = mirror normal velocity
        QP[4][wall] .= Qf[4][wall]

        # impose dual to stress mirror states at symmetry
        QP[2][symmetry] .= Qf[2][symmetry]
        QP[3][symmetry] .= Qf[3][symmetry]
        QP[4][symmetry] .= Qf[4][symmetry]

        # "do nothing" BCs
        QP[2][outflow] .= Qf[2][outflow]
        QP[3][outflow] .= Qf[3][outflow]
        QP[4][outflow] .= β_∞ #Qf[4][outflow]
    end
    # evars = [??, u/T, v/T, -1/T]
    function impose_BCs_entropyvars!(VUP,VUf,md::MeshData)

        # impose freestream at inflow
        V_∞ = v_ufun(primitive_to_conservative(freestream_primvars()...)...)
        for fld = 1:4
            VUP[fld][inflow] .= V_∞[fld]
        end

        # impose mirror states (no-normal flow) at wall (has normal (0,1))
        VUP[2][wall] .= -VUf[2][wall]  # xvelocity = tangential
        VUP[3][wall] .= -VUf[3][wall] # yvelocity = normal
        VUP[4][wall] .= VUf[4][wall]

        # impose dual to stress mirror states at symmetry
        VUP[2][symmetry] .= VUf[2][symmetry]
        VUP[3][symmetry] .= VUf[3][symmetry]
        VUP[4][symmetry] .= VUf[4][symmetry]

        # "do nothing" BCs
        VUP[2][outflow] .= VUf[2][outflow]
        VUP[3][outflow] .= VUf[3][outflow]
        VUP[4][outflow] .= V_∞[4]
    end
    function impose_BCs_stress!(σxP,σyP,σxf,σyf,md::MeshData)
        # zero normal stress at inflow
        for fld = 1:4
            σxP[fld][inflow] .= -σxf[fld][inflow]
        end

        # wall and symmetry = normal vector is (0,1), so normal stress = σy
        # wall = zero Dirichlet condition on velocity
        σyP[2][wall] .= σyf[2][wall]
        σyP[3][wall] .= σyf[3][wall]
        σyP[4][wall] .= -σyf[4][wall]

        # symmetry = zero normal Neumann condition
        σyP[2][symmetry] .= -σyf[2][symmetry]
        σyP[3][symmetry] .= -σyf[3][symmetry]
        σyP[4][symmetry] .= -σyf[4][symmetry]

        # outflow
        σxP[2][outflow] .= σxf[2][outflow]
        σxP[3][outflow] .= σxf[3][outflow]
        σxP[4][outflow] .= σxf[4][outflow]
    end
    return impose_BCs_Qvars!,impose_BCs_entropyvars!,impose_BCs_stress!
end
impose_BCs_Qvars!,impose_BCs_entropyvars!,impose_BCs_stress! = init_BC_funs(md)

function visc_rhs(Q,md::MeshData,rd::RefElemData)

    @unpack Pq,Vq,Vf,LIFT = rd
    @unpack K = md

    Nfields = length(Q)

    # entropy var projection
    VU = v_ufun((x->Vq*x).(Q)...)
    VU = (x->Pq*x).(VU)

    # compute and interpolate to quadrature
    VUf = (x->Vf*x).(VU)
    VUP = (x->x[mapP]).(VUf)
    impose_BCs_entropyvars!(VUP,VUf,md)

    VUx,VUy = dg_grad(VU,VUf,VUP,md,rd)
    VUx = (x->Vq*x).(VUx)
    VUy = (x->Vq*x).(VUy)
    VUq = (x->Vq*x).(VU)

    # initialize sigma_x,sigma_y = viscous rhs
    sigma_x = zero.(VU)
    sigma_y = zero.(VU)
    Kxx,Kxy,Kyy = ntuple(x->MMatrix{4,4}(zeros(Nfields,Nfields)),3)
    sigma_x_e = zero.(getindex.(VUq,:,1))
    sigma_y_e = zero.(getindex.(VUq,:,1))
    for e = 1:K
        fill!.(sigma_x_e,0.0)
        fill!.(sigma_y_e,0.0)

        # mult by matrices and perform local projections
        for i = 1:size(Vq,1)
            vxi = getindex.(VUx,i,e)
            vyi = getindex.(VUy,i,e)
            viscous_matrices!(Kxx,Kxy,Kyy,getindex.(VUq,i,e))

            for col = 1:Nfields
                vxi_col = vxi[col]
                vyi_col = vyi[col]
                for row = 1:Nfields
                    sigma_x_e[row][i] += Kxx[row,col]*vxi_col + Kxy[row,col]*vyi_col
                    sigma_y_e[row][i] += Kxy[col,row]*vxi_col + Kxy[row,col]*vyi_col
                end
            end
        end
        setindex!.(sigma_x,(x->Pq*x).(sigma_x_e),:,e)
        setindex!.(sigma_y,(x->Pq*x).(sigma_y_e),:,e)
    end

    sxf = (x->Vf*x).(sigma_x)
    syf = (x->Vf*x).(sigma_y)
    sxP = (x->x[mapP]).(sxf)
    syP = (x->x[mapP]).(syf)
    impose_BCs_stress!(sxP,syP,sxf,syf,md)

    # add penalty
    tau = .5
    dV = ((xP,x)->xP-x).(VUP,VUf)
    return dg_div(sigma_x,sxf,sxP,sigma_y,syf,syP,md,rd) .+ (x->tau*LIFT*x).(dV)
end

resQ = zero.(Q)
interval = 5
@unpack Vp = rd
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0,markersize=2)
xp = Vp*x
yp = Vp*y

@gif for i = 1:Nsteps

    rhstest = 0
    for INTRK = 1:5
        rhsQ = rhs(Q,md,ops,euler_fluxes)
        visc_rhsQ = visc_rhs(Q,md,rd)

        bcopy!.(rhsQ,@. rhsQ + visc_rhsQ)

        let VhP=VhP,Vq=rd.Vq,wJq=md.wJq
            if INTRK==5
                VU = v_ufun((x->Vq*x).(Q)...)
                VU = (x->VhP*x).(VU)
                for fld in eachindex(rhsQ)
                    VUq = VU[fld][1:size(Vq,1),:]
                    rhstest += sum(wJq.*VUq.*(Vq*rhsQ[fld]))
                end
            end
        end

        # RK step
        bcopy!.(resQ, @. rk4a[INTRK]*resQ + dt*rhsQ)
        bcopy!.(Q, @. Q + rk4b[INTRK]*resQ)
    end

    if i%interval==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")

        vv = Vp*Q[1]
        scatter(xp,yp,vv,zcolor=vv,camera=(0,90),title="Step $i: min/max = $(minimum(vv)), $(maximum(vv))")
    end
end every interval


# #plotting nodes
# @unpack Vp = rd
# gr(aspect_ratio=:equal,legend=false,
#    markerstrokewidth=0,markersize=2)
#

p = pfun(Q...)
ρ = Q[1]
c = @. sqrt(γ*p/ρ)
vv = Vp*(1 ./ c)
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90))
