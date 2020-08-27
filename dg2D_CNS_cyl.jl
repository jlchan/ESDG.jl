using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using StaticArrays # for MArrays
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DTri
using UniformTriMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

"Approximation parameters"
N = 3           # order of approximation
CFL = .1      # sets dt0
T = 1.0

"Viscous parameters"
mu = 1e-4
lambda = -2/3*mu
Pr = .71

poly = polygon_unitSquareWithHole()
poly.point .= @. poly.point*2 - 1
poly.point[:,1:4] .*= 2.5
poly.point[1,2:3] .*= 1.5
poly.hole .= 0.0
mesh = create_mesh(poly, info_str="CNS hole", voronoi=true, delaunay=true, set_area_max=true)
VX = mesh.point[1,:]
VY = mesh.point[2,:]
EToV = Matrix(mesh.cell')

# "Mesh related variables"
# K1D = 16
# Kx = K1D
# Ky = K1D
# VX, VY, EToV = uniform_tri_mesh(Kx, Ky)
# @. VX = 2*VX
# @. VY = 2*VY

# initialize ref element and mesh
# specify lobatto nodes to automatically get DG-SEM mass lumping
rd = init_reference_tri(N)
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
    Ma = 1.5
    return 1,1,0,1/(Ma^2*γ)
end
U = ntuple(a->zeros(size(x)),4)
for field = 1:4
    U[field] .= freestream_primvars()[field]
end
Q = primitive_to_conservative(U...)

# rho = @. 1.0 + exp(-10*(x^2+y^2))
# u = zeros(size(x))
# v = zeros(size(x))
# p = @. rho^γ
# Q = primitive_to_conservative(rho,u,v,p)

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
h  = minimum(minimum(md.J,dims=1)./maximum(md.sJ,dims=1)) # J/sJ = O(h)
dt0 = CFL * h / CN
dt = dt0
# Nsteps = convert(Int,ceil(T/dt))
# dt = T/Nsteps

"dense version - speed up by prealloc + transpose for col major "
function dense_hadamard_sum(Qhe,ops,vgeo,flux_fun,Nq=Inf)

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

            if (i > Nq & j > Nq)==false # skip over zero blocks
                Fx,Fy = flux_fun(Qi,Qj,Qlogi,Qlogj)
                @. QFi += QxTr[j,i]*Fx + QyTr[j,i]*Fy
            end
        end

        for field in eachindex(Qhe)
            QF[field][i] = QFi[field]
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
    impose_BCs_inviscid!(QP,QM,md)

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
        QFe = dense_hadamard_sum(Qhe,Qops,vgeo_local,flux_fun,Nq)

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
    surf(uP,uf,n) = LIFT*(@. .5*(uP-uf)*n)
    rhsx = volx.(Qr,Qs) .+ surf.(QP,Qf,nxJ)
    rhsy = voly.(Qr,Qs) .+ surf.(QP,Qf,nyJ)
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
    @unpack xf,yf,mapP,mapB,nxJ,nyJ,sJ = md
    xb,yb = (x->x[mapB]).((xf,yf))

    freestream   = mapB[findall((@. xb^2 + yb^2 > 2))]
    # freestream = mapB[findall((@. xb^2 + yb^2 < 2))] # ignore
    wall         = mapB[findall((@. xb^2 + yb^2 < 2))]

    nxw = nxJ[wall]./sJ[wall]
    nyw = nyJ[wall]./sJ[wall]

    nxfree = nxJ[freestream]./sJ[freestream]
    nyfree = nyJ[freestream]./sJ[freestream]

    # let nxb=nxb,nyb=nyb,freestream=freestream,wall=wall

    function impose_BCs_inviscid!(QP,Qf,md::MeshData)
        ρ_∞,u_∞,v_∞,p_∞ = freestream_primvars()
        β_∞ = betafun(primitive_to_conservative(ρ_∞,u_∞,v_∞,p_∞)...)
        QP[1][freestream] .= ρ_∞
        QP[2][freestream] .= u_∞
        QP[3][freestream] .= v_∞
        QP[4][freestream] .= β_∞

        # impose mirror states (no-normal flow) at wall (has normal (0,1))
        u = Qf[2][wall]
        v = Qf[3][wall]
        Un = @. u*nxw + v*nyw
        QP[2][wall] .= @. u - 2*Un*nxw
        QP[3][wall] .= @. v - 2*Un*nyw
        QP[4][wall] .= Qf[4][wall]
    end
    # evars = [??, u/T, v/T, -1/T]
    function impose_BCs_entropyvars!(VUP,VUf,md::MeshData)

        # impose freestream at inflow
        V_∞ = v_ufun(primitive_to_conservative(freestream_primvars()...)...)
        for field = 1:4
            VUP[field][freestream] .= V_∞[field]
        end

        # impose mirror states (no-normal flow) at wall (has normal (0,1))
        VUP[2][wall] .= -VUf[2][wall]  # xvelocity = tangential
        VUP[3][wall] .= -VUf[3][wall] # yvelocity = normal
        VUP[4][wall] .= VUf[4][wall]

    end
    function impose_BCs_stress!(σxP,σyP,σxf,σyf,md::MeshData)

        # zero normal stress at inflow
        for field = 1:4
            σn = @. σxf[field][freestream]*nxfree + σyf[field][freestream]*nyfree
            σxP[field][freestream] .= @. σxf[field][freestream] - 2*σn*nxfree
            σyP[field][freestream] .= @. σyf[field][freestream] - 2*σn*nyfree
        end

        # wall and symmetry = normal vector is (0,1), so normal stress = σy
        # wall = zero Dirichlet condition on velocity
        σyP[2][wall] .= σyf[2][wall]
        σyP[3][wall] .= σyf[3][wall]
        σ4n = @. σxf[4][wall]*nxw + σyf[4][wall]*nyw
        σxP[4][wall] .= @. σxf[4][wall] - 2*σ4n*nxw
        σyP[4][wall] .= @. σyf[4][wall] - 2*σ4n*nyw
    end
    return impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress!
end
impose_BCs_inviscid!,impose_BCs_entropyvars!,impose_BCs_stress! = init_BC_funs(md)

function visc_rhs(Q,md::MeshData,rd::RefElemData)

    @unpack Pq,Vq,Vf,LIFT = rd
    @unpack K,mapP,J,sJ,nxJ,nyJ = md

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

    # # add penalty
    dV = ((xP,x)->(xP-x)).(VUP,VUf)

    # # arbitrary scaling
    # VUq = v_ufun((x->Vq*x).(Q)...)
    # VUf = v_ufun((x->Vf*x).(Q)...)
    # VUf_proj = (x->Vf*Pq*x).(VUq)
    # Nfaces = Int(3)
    # Nfq = Int(length(rd.wf)/Nfaces)
    # face_norms(x) = sum(reshape(wsJ.*x.^2,Nfq,rd.Nfaces*md.K),dims=1)
    # VUf_err = vec(sum(face_norms.(VUf.-VUf_proj))./sum(face_norms.(VUf_proj)))
    # tau = 100*min(max(maximum(VUf_err),1/10),1) # global penalty scaling

    dV[1] .= 0 # ignore penalty for mass equation
    # tau = .1
    # LdV = (x->nxJ.*(Vf*((LIFT*(@. x*nxJ))./J)) + nyJ.*(Vf*((LIFT*(@. x*nyJ))./J))).(dV)
    # visc_penalty = (x-> -tau*LIFT*(x[mapP]-x)).(LdV) # BR2 penalty
    h  = minimum(minimum(md.J,dims=1)./maximum(md.sJ,dims=1))
    tau = .25/h
    visc_penalty = (x->tau*(LIFT*(x.*md.sJ))./md.J).(dV) # IPDG type penalty

    return dg_div(sigma_x,sxf,sxP,sigma_y,syf,syP,md,rd) .+ visc_penalty
end


function rhsRK(Q,rd,md,ops,fluxes::Fxn) where Fxn
    rhsQ = rhs(Q,md,ops,euler_fluxes)
    visc_rhsQ = visc_rhs(Q,md,rd)
    bcopy!.(rhsQ, @. rhsQ + visc_rhsQ)

    let Pq=rd.Pq,Vq=rd.Vq,wJq=md.wJq
        rhstest = 0.0
        VU = v_ufun((x->Vq*x).(Q)...)
        VUq = (x->Vq*Pq*x).(VU)
        for field in eachindex(rhsQ)
            rhstest += sum(wJq.*VUq[field].*(Vq*rhsQ[field]))
        end
        return rhsQ,rhstest
    end
end

# dopri coeffs
rka,rkE,rkc = dopri45_coeffs()

# DOPRI storage
Qtmp = similar.(Q)
rhsQrk = ntuple(x->zero.(Q),length(rkE))

errEst = 0.0
prevErrEst = 0.0

t = 0.0
i = 0
interval = 10

dthist = [dt]
thist = [0.0]
errhist = [0.0]
wsJ = diagm(rd.wf)*md.sJ

rhsQ,_ = rhsRK(Q,rd,md,ops,euler_fluxes)
bcopy!.(rhsQrk[1],rhsQ) # initialize DOPRI rhs (FSAL property)
while t < T
    # DOPRI step and
    rhstest = 0.0
    for INTRK = 2:7
        k = zero.(Qtmp)
        for s = 1:INTRK-1
            bcopy!.(k, @. k + rka[INTRK,s]*rhsQrk[s])
        end
        bcopy!.(Qtmp, @. Q + dt*k)
        rhsQ,rhstest = rhsRK(Qtmp,rd,md,ops,euler_fluxes)
        bcopy!.(rhsQrk[INTRK],rhsQ)
    end
    errEstVec = zero.(Qtmp)
    for s = 1:7
        bcopy!.(errEstVec, @. errEstVec + rkE[s]*rhsQrk[s])
    end

    errTol = 1e-5
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

    global i = i + 1  # number of total steps attempted
    if i%interval==0
        println("i = $i, t = $t, dt = $dtnew, errEst = $errEst, rhstest = $rhstest")
    end
end

scatter!(thist,dthist)
# scatter!(thist,errhist./maximum(errhist))

#plotting nodes
@unpack Vp = rd
gr(aspect_ratio=:equal,legend=false,
   markerstrokewidth=0)

vv = Vp*Q[1]
scatter(Vp*x,Vp*y,vv,zcolor=vv,camera=(0,90),ms=2)
