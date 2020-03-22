# "Packages"
using Revise # reduce need for recompile
using Plots
using Documenter
using LinearAlgebra
using SparseArrays
using UnPack

# "User defined modules"
push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad # face trace space
using Basis3DHex
using UniformHexMesh

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

using SetupDG

N = 2
K1D = 8
T = 2/3 # endtime
CFL = .75

VX,VY,VZ,EToV = uniform_hex_mesh(K1D,K1D,K1D)

rd = init_reference_hex(N,gauss_lobatto_quad(0,0,N))
md = init_mesh((VX,VY,VZ),EToV,rd)

# Make hybridized SBP operators
@unpack M,Dr,Ds,Dt,Pq,Vq,Vf,nrJ,nsJ,ntJ,wf = rd
Qr = Pq'*M*Dr*Pq
Qs = Pq'*M*Ds*Pq
Qt = Pq'*M*Dt*Pq
Ef = Vf*Pq
Br = diagm(wf.*nrJ)
Bs = diagm(wf.*nsJ)
Bt = diagm(wf.*ntJ)
Qrh = .5*[Qr-Qr' Ef'*Br;
        -Br*Ef Br]
Qsh = .5*[Qs-Qs' Ef'*Bs;
        -Bs*Ef Bs]
Qth = .5*[Qt-Qt' Ef'*Bt;
        -Bt*Ef Bt]

"sparse skew symmetric versions of the operators"
Qrhskew = .5*(Qrh-transpose(Qrh))
Qshskew = .5*(Qsh-transpose(Qsh))
Qthskew = .5*(Qth-transpose(Qth))
Qrhskew_sparse = droptol!(sparse(Qrhskew),1e-12)
Qshskew_sparse = droptol!(sparse(Qshskew),1e-12)
Qthskew_sparse = droptol!(sparse(Qthskew),1e-12)

# precompute union of sparse ids for Qr, Qs
Qnzids = [unique([Qrhskew_sparse[i,:].nzind; Qshskew_sparse[i,:].nzind; Qthskew_sparse[i,:].nzind]) for i = 1:size(Qrhskew,1)]

# make periodic
@unpack Nfaces = rd
@unpack xf,yf,zf,K,mapM,mapP,mapB = md
LX = 2; LY = 2; LZ = 2
mapPB = build_periodic_boundary_maps(xf,yf,zf,LX,LY,LZ,Nfaces*K,mapM,mapP,mapB)
mapP[mapB] = mapPB
@pack! md = mapP

# add curved mapping
@unpack x,y,z = md
a = .05
dx = @. (x-1)*(x+1)*(y-1)*(y+1)*(z-1)*(z+1)
x = x + a.*dx
y = y + a.*dx
z = z + a.*dx

# recompute phys nodes, geofacs, etc for a curved mesh
@unpack xq,yq,zq = md
xq,yq,zq = (x->Vq*x).((x,y,z))
@pack! md = xq,yq,zq

vgeo = geometric_factors(x,y,z,Dr,Ds,Dt)
rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ,J = vgeo
nxJ = nrJ.*(Vf*rxJ) + nsJ.*(Vf*sxJ) + ntJ.*(Vf*txJ)
nyJ = nrJ.*(Vf*ryJ) + nsJ.*(Vf*syJ) + ntJ.*(Vf*tyJ)
nzJ = nrJ.*(Vf*rzJ) + nsJ.*(Vf*szJ) + ntJ.*(Vf*tzJ)
sJ = @. sqrt(nxJ.^2 + nyJ.^2 + nzJ.^2)
@pack! md = nxJ,nyJ,nzJ,sJ

# interp geofacs to both vol/surface quadrature points
rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ = (x->[Vq;Vf]*x).(vgeo)
@pack! md = rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ

# convert to quadrature node basis
@unpack wq=rd
Vh = droptol!(sparse(vcat(diagm(ones(length(wq))), Ef)),1e-12)
Ph = droptol!(sparse(2*diagm(@. 1/wq)*transpose(Vh)),1e-12)
Lf = droptol!(sparse(diagm(@. 1/wq)*(transpose(Ef)*diagm(wf))),1e-12)
ops = (Qrhskew_sparse,Qshskew_sparse,Qthskew_sparse,Qnzids,Ph,Lf)
J = Vq*J
wJq = diagm(wq)*J # recompute for curved
@pack! md = J,wJq

# initial conditions
rhoex(x,y,z,t) = @. 2 + .5*sin(pi*(x-t))
rho = rhoex(xq,yq,zq,0)
u = ones(size(x))
v = zeros(size(x))
w = zeros(size(x))
p = ones(size(x))
Q = primitive_to_conservative(rho,u,v,w,p)

"timestepping"
rk4a,rk4b,rk4c = rk45_coeffs()
CN = (N+1)*(N+2)*3/2  # estimated trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"sparse version - precompute sparse row ids for speed"
function sparse_hadamard_sum(Qhe,ops,vgeo,flux_fun)

    (Qr,Qs,Qt,Qnzids) = ops
    nrows = size(Qr,1)
    nfields = length(Qhe)

    # precompute logs for logmean
    (rho,u,v,w,beta) = Qhe
    Qlog = (log.(rho), log.(beta))

    rhsQe = ntuple(x->zeros(nrows),nfields)
    rhsi = zeros(nfields) # prealloc a small array for accumulation
    for i = 1:nrows
        Qi = (x->x[i]).(Qhe)
        Qlogi = (x->x[i]).(Qlog)
        vgeo_i = (x->x[i]).(vgeo)

        fill!(rhsi,0) # reset rhsi before accumulation
        for j = Qnzids[i] # nonzero row entries
            Qj = (x->x[j]).(Qhe)
            Qlogj = (x->x[j]).(Qlog)
            vgeo_j = (x->x[j]).(vgeo)

            avg(uL,uR) = .5*(uL+uR)
            rxJa,sxJa,txJa,ryJa,syJa,tyJa,rzJa,szJa,tzJa = avg.(vgeo_i,vgeo_j)

            Fx,Fy,Fz = flux_fun(Qi,Qj,Qlogi,Qlogj)
            Fr = @. rxJa*Fx + ryJa*Fy + rzJa*Fz
            Fs = @. sxJa*Fx + syJa*Fy + szJa*Fz
            Ft = @. txJa*Fx + tyJa*Fy + tzJa*Fz

            # sum(Qx.*Fx + Qy.*Fy,2) = sum(Qr*rxJ*Fx + Qs*sxJ*Fx + Qr*ryJ*Fy ...)
            @. rhsi += Qr[i,j]*Fr + Qs[i,j]*Fs + Qt[i,j]*Ft
        end

        # faster than one-line fixes (no return args)
        for fld in eachindex(rhsQe)
            rhsQe[fld][i] = rhsi[fld]
        end
    end

    return rhsQe
end

function rhs(Q,md::MeshData,ops,flux_fun,compute_rhstest=false)

    # unpack args
    (Qrh_sparse,Qsh_sparse,Qth_sparse,Qnzids,Ph,Lf)=ops
    @unpack rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ=md
    @unpack J,wJq,nxJ,nyJ,nzJ,sJ,mapP,mapB,K = md
    Nq,Nh = size(Ph)

    # entropy projection
    VU = v_ufun(Q...)
    Uf = u_vfun((x->Ef*x).(VU)...) # conservative vars
    Uh = vcat.(Q,Uf)

    # convert to rho,u,v,beta vars
    (rho,rhou,rhov,rhow,E) = Uh
    beta = betafun(rho,rhou,rhov,rhow,E)
    Qh = (rho,rhou./rho,rhov./rho,rhow./rho,beta) # redefine Q = (rho,U,β)

    QM = (x->x[Nq+1:end,:]).(Qh)
    QP = (x->x[mapP]).(QM)

    # lax friedrichs dissipation
    (rho,rhou,rhov,rhow,E) = Uf
    rhoU_n = @. (rhou*nxJ + rhov*nyJ + rhow*nzJ)/sJ
    lam = abs.(wavespeed(rho,rhoU_n,E))
    LFc = .5*max.(lam,lam[mapP]).*sJ

    fSx,fSy,fSz = flux_fun(QM,QP)
    normal_flux(fx,fy,fz,uM) = fx.*nxJ + fy.*nyJ + fz.*nzJ - LFc.*(uM[mapP]-uM)
    flux = normal_flux.(fSx,fSy,fSz,Uf)
    # flux = ntuple(x->zeros(size(mapP)),length(QM))
    # for i = 1:length(mapP)
    #     QMi = (x->x[i]).(QM)
    #     QPi = (x->x[i]).(QP)
    #     fSx,fSy,fSz = flux_fun(QMi,QPi)
    #     normal_flux!(f,fx,fy,fz,u) = f[i] = fx*nxJ[i] + fy*nyJ[i] + fz*nzJ[i] - LFc[i]*(u[mapP[i]]-u[i])
    #     normal_flux!.(flux,fSx,fSy,fSz,Uf)
    # end
    rhsQ = (x->Lf*x).(flux)

    # compute volume contributions using flux differencing
    for e = 1:K
        Qhe = tuple((x->x[:,e]).(Qh)...)
        vgeo_elem = (x->x[:,e]).((rxJ,sxJ,txJ,ryJ,syJ,tyJ,rzJ,szJ,tzJ)) # assumes curved elements

        Qops = (Qrh_sparse,Qsh_sparse,Qth_sparse,Qnzids)
        QFe = sparse_hadamard_sum(Qhe,Qops,vgeo_elem,flux_fun) # sum(Q.*F,dims=2)

        mxm_accum!(X,x) = X[:,e] += Ph*x
        mxm_accum!.(rhsQ,QFe)
    end

    rhsQ = (x -> -x./J).(rhsQ)

    rhstest = 0
    if compute_rhstest
        for fld in eachindex(rhsQ)
            rhstest += sum(wJq.*VU[fld].*rhsQ[fld])
        end
    end

    return rhsQ,rhstest # scale by Jacobian
end


# force Q to be an array of arrays for mutability
Q = collect(Q)
resQ = [zeros(size(x)) for i in eachindex(Q)]
for i = 1:Nsteps

    rhstest = 0
    for INTRK = 1:5
        compute_rhstest = INTRK==5
        rhsQ,rhstest = rhs(Q,md,ops,euler_fluxes,compute_rhstest)

        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q += rk4b[INTRK]*resQ
    end

    if i%10==0 || i==Nsteps
        println("Time step: $i out of $Nsteps with rhstest = $rhstest")
    end
end

(rho,rhou,rhov,rhow,E) = (x->Pq*x).(Q) # project back to Lobatto nodes

@unpack VDM=rd
rq2,sq2,tq2,wq2 = quad_nodes_3D(N+2)
Vq2 = vandermonde_3D(N,rq2,sq2,tq2)/VDM
(xq2,yq2,zq2) = (x->Vq2*x).((x,y,z))
wJq2 = abs.(diagm(wq2)*(Vq2*Pq*J)) # recall J = converted to quad node basis
L2err = sqrt(sum(wJq2.*(Vq2*rho - rhoex(xq2,yq2,zq2,T)).^2))
# L2err = sqrt(sum(wJq.*(Vq*rho - rhoex(xq,yq,zq,T)).^2))
@show L2err

# gr(aspect_ratio=1,legend=false,
#    markerstrokewidth=0,markersize=2)
#
# @unpack Vp = rd
# (xp,yp,zp,vv) = (x->Vp*x).((x,y,z,rho))
#
# ids = map(x->x[1],findall(@. abs(zp[:])<1e-10))
# (xp,yp,zp,vv) = (x->x[ids]).((xp,yp,zp,vv))
# scatter(xp,yp,vv,zcolor=vv,camera=(0,90))
