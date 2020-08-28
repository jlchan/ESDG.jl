using Revise # reduce recompilation time
using Plots
using LinearAlgebra
using StaticArrays
using SparseArrays
using BenchmarkTools
using UnPack
using ForwardDiff

push!(LOAD_PATH, "./src")
using CommonUtils
using NodesAndModes
using NodesAndModes.Tri
using UniformMeshes
using SetupDG
using ExplicitFluxDiffJacobians

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

"Approximation parameters"
N = 2 # The order of approximation
K1D = 8
CFL = 1
T = 1.0 # endtime

"Mesh related variables"
VX, VY, EToV = uniform_tri_mesh(3*K1D,2*K1D)
# VX = @. VX - .3*sin(pi*VX)
VX = @. (1+VX)/2 * 15
VY = @. VY*5

# initialize ref element and mesh
rd = init_reference_tri(N)
md = init_mesh((VX,VY),EToV,rd)

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,FToF,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps!(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB,FToF)
mapP[mapB] = mapPB

## construct hybridized SBP operators

function build_SBP_ops(rd::RefElemData)
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

        # make skew symmetric versions of the operators"
        Qrhskew = .5*(Qrh-transpose(Qrh))
        Qshskew = .5*(Qsh-transpose(Qsh))

        return Qrhskew,Qshskew,Vh,Ph,VhP,M
end

# interpolate geofacs to both vol/surf nodes
@unpack Vq,Vf = rd
@unpack rxJ, sxJ, ryJ, syJ = md
rxJ, sxJ, ryJ, syJ = (x->[Vq;Vf]*x).((rxJ, sxJ, ryJ, syJ)) # interp to hybridized points

## global matrices

function build_global_ops(rd::RefElemData,md::MeshData)
        Qrhskew,Qshskew,Vh,Ph,VhP,M = build_SBP_ops(rd)

        Ax,Ay,Bx,By,B = assemble_global_SBP_matrices_2D(rd,md,Qrhskew,Qshskew)

        # add off-diagonal couplings
        Ax += Bx
        Ay += By

        Ax *= 2 # for flux differencing
        Ay *= 2

        AxTr = sparse(transpose(Ax))
        AyTr = sparse(transpose(Ay))
        Bx   = abs.(Bx) # create LF penalization term

        # globalize operators and nodes
        @unpack x,y,J = md
        VhTr = kron(speye(K),sparse(transpose(Vh)))
        Vh   = kron(speye(K),sparse(Vh))
        M    = kron(spdiagm(0 => J[1,:]),sparse(M))
        Vq   = kron(speye(K),sparse(rd.Vq))
        Pq   = kron(speye(K),sparse(rd.Pq))
        Ph   = kron(spdiagm(0 => 1 ./ J[1,:]), sparse(Ph))
        x,y = (a->a[:]).((x,y))

        return Ax,Ay,AxTr,AyTr,Bx,B,VhTr,Vh,M,Ph,Vq,Pq,x,y
end

Ax,Ay,AxTr,AyTr,Bx,B,VhTr,Vh,M,Ph,Vq,Pq,x,y =
        build_global_ops(rd::RefElemData,md::MeshData)
println("Done building global ops")

# jacspy = I+Ax+Ay
# jac = droptol!(jacspy,1e-12)
# jac = droptol!(VhTr*jacspy*Vh,1e-12)
# display(spy(jac .!= 0,ms=2.25))
# # title="nnz = $(nnz(jac))",
# error("d")
## EC and dissipative fluxes

function LF(uL,uR,nxL,nyL,nxR,nyR)
        rhoL,rhouL,rhovL,EL = uL
        rhoR,rhouR,rhovR,ER = uR

        nx = nxL #.5*(nxL+nxR)
        ny = nyL #.5*(nyL+nyR)
        rhoUnL = @. rhouL*nx + rhovL*ny
        rhoUnR = @. rhouR*nx + rhovR*ny
        cL   = @. wavespeed(rhoL,rhoUnL,EL)
        cR   = @. wavespeed(rhoR,rhoUnR,ER)
        lam  = @. sqrt(.5*(cL^2+cR^2)) # arith avg is type stable with ForwardDiff
        return (@. lam*(uL-uR))
end


function initFxns()
        function UtoQ(rho,rhou,rhov,E)
            beta = betafun(rho,rhou,rhov,E)
            # beta = betafun(U...)
            return (rho,rhou./rho,rhov./rho,beta),(log.(rho),log.(beta))
        end
        function Fx(UL,UR)
        # ForwardDiff behaves better without splatting -
        # see https://github.com/JuliaDiff/ForwardDiff.jl/issues/89
            QL,QlogL = UtoQ(UL[1],UL[2],UL[3],UL[4])
            QR,QlogR = UtoQ(UR[1],UR[2],UR[3],UR[4])
            Fx1,Fx2,Fx3,Fx4 = euler_flux_x(QL...,QR...,QlogL...,QlogR...)
            return SVector{4}(Fx1,Fx2,Fx3,Fx4)
        end
        function Fy(UL,UR)
            QL,QlogL = UtoQ(UL[1],UL[2],UL[3],UL[4])
            QR,QlogR = UtoQ(UR[1],UR[2],UR[3],UR[4])
            Fy1,Fy2,Fy3,Fy4 = euler_flux_y(QL...,QR...,QlogL...,QlogR...)
            return SVector{4}(Fy1,Fy2,Fy3,Fy4)
        end
        return Fx,Fy
end
Fx,Fy = initFxns()

# AD for jacobians
function initJacobian(F::Fxn,uR) where Fxn
    cfg = ForwardDiff.JacobianConfig(F, uR, ForwardDiff.Chunk{4}())
    out = zeros(eltype(uR),length(uR),length(uR))

    function df!(out,uL,uR,args...)
        ForwardDiff.jacobian!(out, uR->F(uL,uR,args...), uR, cfg)
        return out
    end
    function df(uL,uR,args...)
        ForwardDiff.jacobian!(out, uR->F(uL,uR,args...), uR, cfg)
        return out
    end
    return df,df!
end

dFx_cfg,dFx_cfg! = initJacobian(Fx,SVector{4}(zeros(Float64,4)...))
dFy_cfg,_ = initJacobian(Fy,SVector{4}(zeros(Float64,4)...))
dLF_cfg,_ = initJacobian(LF,SVector{4}(zeros(Float64,4)...))
dFx(uL,uR) = ForwardDiff.jacobian(uR->Fx(uL,uR),uR)
dFy(uL,uR) = ForwardDiff.jacobian(uR->Fy(uL,uR),uR)
dLF(uL,uR,args...) = ForwardDiff.jacobian(uR->LF(uL,uR,args...),uR)

## mappings between conservative and entropy variables and vice vera

# dVdU_fun(U) = ForwardDiff.jacobian(U->SVector(v_ufun(U...)...),U)
# dUdV_fun(V) = ForwardDiff.jacobian(V->SVector(u_vfun(V...)...),V)
dVdU_fun(U) = dVdU_explicit(U...)
dUdV_fun(V) = dUdV_explicit(V...)

## nonlinear solver setup - uses closure to initialize arrays and other things
function init_newton_fxn(Q,ops,rd::RefElemData,md::MeshData,funs,dt)

        # set up normals for use in penalty term
        @unpack nxJ,nyJ,sJ = md
        Nq,Np = size(rd.Vq)
        Nf    = size(rd.Vf,1)
        Nh    = Nq + Nf
        fids  = Nq + 1:Nh
        nxhmat,nyhmat = ntuple(x->zeros(Nh,K),2)
        nxhmat[fids,:] = @. nxJ/sJ
        nyhmat[fids,:] = @. nyJ/sJ
        nxh,nyh = vec.((nxhmat,nyhmat))

        # get lengths of arrays
        Ax,Ay,AxTr,AyTr,Bx,B,M,Vh,Ph,Vq,Pq = ops
        Fx,Fy,dFx,dFy,LF,dLF = funs

        Nfields = length(Q)
        Id_fld  = speye(Nfields) # for Kronecker expansion to large matrices - fix later with lazy evals
        Vq_fld  = droptol!(kron(Id_fld,Vq),1e-13)
        VhP     = Vh*Pq
        VhP_fld = droptol!(kron(Id_fld,VhP),1e-13)
        Vh_fld  = droptol!(kron(Id_fld,Vh),1e-13)
        M_fld   = droptol!(kron(Id_fld,M),1e-13)

        # init jacobian matrix (no need for entropy projection since we'll zero it out later)
        dFdU_h = repeat(I+Ax+Ay,Nfields,Nfields)
        dVdU_q = repeat(speye(size(Vq,1)),Nfields,Nfields)
        dUdV_h = repeat(speye(size(Vh,1)),Nfields,Nfields)
        function midpt_newton_iter!(Qnew, Qprev) # for Burgers' eqn specifically

                # perform entropy projection
                Uq    = SVector((x->Vq*x).(Qnew))
                # VUh   = SVector((x->VhP*x).(v_ufun(Uq...)))
                # Qh    = SVector(u_vfun(VUh...))
                VUh   = SVector((x->VhP*x).(v_ufun(Uq[1],Uq[2],Uq[3],Uq[4]))) # why is this type stable and splatting isn't?
                Qh    = SVector(u_vfun(VUh[1],VUh[2],VUh[3],VUh[4]))

                ftmp  = hadamard_sum(AxTr,Fx,Qh) + hadamard_sum(AyTr,Fy,Qh) + hadamard_sum(B,LF,Qh,nxh,nyh)
                f     = vcat((x->Ph*x).(ftmp)...)
                res   = vcat(Qnew...) + .5*dt*f - vcat(Qprev...)

                # compute RHStest for entropy checks
                VUproj = (x->Pq*x).(v_ufun(Uq[1],Uq[2],Uq[3],Uq[4]))
                Mrhs = (x->Vh'*x).(ftmp)
                rhstest = sum(sum(((x,y)->x.*y).(VUproj,Mrhs)))

                fill!(dFdU_h.nzval,0.0)
                accum_hadamard_jacobian!(dFdU_h, Ax, dFx, Qh)
                accum_hadamard_jacobian!(dFdU_h, Ay, dFy, Qh)
                accum_hadamard_jacobian!(dFdU_h, B,  dLF, Qh, nxh, nyh) # flux term involving normals
                banded_matrix_function!(dVdU_q, dVdU_fun, Uq)
                banded_matrix_function!(dUdV_h, dUdV_fun, VUh)
                dFdU   = droptol!(transpose(Vh_fld)*(dFdU_h*dUdV_h*VhP_fld*dVdU_q*Vq_fld),1e-13)

                # solve and update
                dQ   = (M_fld + .5*dt*dFdU)\(M_fld*res)
                dQnorm = norm(dQ)/sum(norm.(Qprev))
                Qtmp = reshape(vcat(Qnew...) - dQ, length(Q[1]), Nfields)   # convert Qnew to column vector for update

                # return columnize(reshape(Qtmp,length(Q[1]),Nfields)),dQnorm # convert back to array of arrays
                for fld = 1:length(Qnew)
                        Qnew[fld] .= Qtmp[:,fld]
                end
                return dQnorm,rhstest
        end
        return midpt_newton_iter!
end

## init condition, rhs

@unpack xq,yq = md

# rho = vec(rd.Pq*(@. 1 + .1*(abs(xq)<.5).*(abs(yq)<.5)))
# rhou = vec(rd.Pq*(@. .0*(abs(xq)<.5).*(abs(yq)<.5)))
# rhov = vec(rd.Pq*(@. -.0*(abs(xq)<.5).*(abs(yq)<.5)))
# E = @. rho^1.4
# Q = [rho,rhou,rhov,E]

Q = collect(primitive_to_conservative(vortex(x,y,0)...))
Qnew = copy.(Q)

# convert to tuple
Q = tuple(Q...)
Qnew = tuple(Qnew...)

# set time-stepping constants
CN = (N+1)*(N+2)/2  # estimated trace constant
h = minimum(md.J[1,:]./md.sJ[1,:]) # ratio of J/Jf = O(h^d/h^d-1)
dt = CFL * 2 * h / CN
# dt = .1
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# initialize jacobian
ops = (Ax,Ay,copy(transpose(Ax)),copy(transpose(Ay)),Bx,B,M,Vh,Ph,Vq,Pq) # pack inputs together
#funs = (Fx,Fy,dFx,dFy,LF,dLF)
funs = (Fx,Fy,dFx_cfg,dFy_cfg,LF,dLF_cfg)
midpt_newton_iter! = init_newton_fxn(Q,ops,rd,md,funs,dt)

## newton time iteration

bcopy!(utarget,u) = utarget .= u

it_count = zeros(Nsteps)
energy   = zeros(Nsteps)
for i = 1:Nsteps
        # global Q,Qnew

        bcopy!.(Qnew,Q)  # copy / over-write at each timestep
        iter = 0
        dQnorm = 1
        rhstest = 0
        while dQnorm > 1e-11
                # Qnew,dQnorm = midpt_newton_iter!(Qnew,Q)
                dQnorm,rhstest = midpt_newton_iter!(Qnew,Q)
                iter += 1
                if iter > 15
                        println("iter = $iter, ||dQ|| = $dQnorm")
                end
        end
        it_count[i] = iter
        bcopy!.(Q, @. 2*Qnew - Q) # implicit midpoint rule

        Qq = (x->Vq*x).(Q)
        energy[i] = sum(vec(md.wJq).*Sfun(Qq...))

        if i%5==0 || i==Nsteps
                println("Number of time steps $i out of $Nsteps, rhstest = $rhstest")
                # display(scatter(x,Q[1]))
        end
end

@unpack VDM = rd
rp, sp = equi_nodes_2D(25)
Vp = vandermonde_2D(N,rp,sp)/VDM

gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0)
xp,yp,up = (x->Vp*reshape(x,size(Vp,2),K)).((x,y,Q[1]))
# display(scatter(xp,yp,up,zcolor=up,cam=(3,25),axis=false))
scatter(xp,yp,-1e-8*ones(size(up)),zcolor=up,cam=(0,90),border=:none,axis=false,markersize=1)
# png("sol_unif_mesh.png")
for e = 1:K
        vids = EToV[e,:]
        for f = 1:3
                vx = VX[vids[rd.fv[f]]]
                vy = VY[vids[rd.fv[f]]]
                plot!(vx,vy,linewidth=.5,legend=false,linecolor=:black)
        end
end
display(plot!())
@show energy[end]-energy[1]

@unpack xq,yq,wJq = md
Qex = primitive_to_conservative(vortex(vec.((xq,yq))...,T)...)
# err = ((x->Vq*x).(Q) .- Qex)
@show sqrt(sum(sum.((x->vec(wJq).*x.^2).((x->Vq*x).(Q).-vec.(Qex)))))
