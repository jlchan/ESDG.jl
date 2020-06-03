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
using Basis1D
using Basis2DTri
using UniformTriMesh
using SetupDG
using ExplicitJacobians

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

"Approximation parameters"
N = 1 # The order of approximation
K1D = 4
CFL = 1.0
T = 1 # endtime

"Mesh related variables"
VX, VY, EToV = uniform_tri_mesh(K1D)
# VX = @. VX - .3*sin(pi*VX)

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

## define Euler fluxes for 2D
# convert to flux variables

# function F(UL,UR)
#     QL,QlogL = UtoQ(UL)
#     QR,QlogR = UtoQ(UR)
#     Fx,Fy = euler_fluxes(QL...,QR...,QlogL...,QlogR...)
#     # @show Fx,Fy
#     return SVector{4}(Fx),SVector{4}(Fy)
#     # return SVector{4}(Fx...),SVector{4}(Fy...)
# end
# # extract coordinate fluxes
# Fx = (uL,uR)->F(uL,uR)[1]
# Fy = (uL,uR)->F(uL,uR)[2]

function initFxns()
        function UtoQ(U)
            rho,rhou,rhov,E = U
            beta = betafun(U[1],U[2],U[3],U[4])
            # beta = betafun(U...)
            return (rho,rhou./rho,rhov./rho,beta),(log.(rho),log.(beta))
        end
        function Fx(UL,UR)
            QL,QlogL = UtoQ(UL)
            QR,QlogR = UtoQ(UR)
            Fx = euler_flux_x(QL...,QR...,QlogL...,QlogR...)
            return SVector{4}(Fx)
        end
        function Fy(UL,UR)
            QL,QlogL = UtoQ(UL)
            QR,QlogR = UtoQ(UR)
            Fy = euler_flux_y(QL...,QR...,QlogL...,QlogR...)
            return SVector{4}(Fy)
        end
        return Fx,Fy
end
Fx,Fy = initFxns()

# AD for jacobians
dFx(uL,uR) = ForwardDiff.jacobian(uR->Fx(uL,uR),uR)
dFy(uL,uR) = ForwardDiff.jacobian(uR->Fy(uL,uR),uR)

## dissipative flux

function LF(uL,uR,nxL,nyL,nxR,nyR)
        rhoL,rhouL,rhovL,EL = uL
        rhoR,rhouR,rhovR,ER = uR

        nx = nxL #.5*(nxL+nxR)
        ny = nyL #.5*(nyL+nyR)
        rhoUnL = @. rhouL*nx + rhovL*ny
        rhoUnR = @. rhouR*nx + rhovR*ny
        cL   = @. wavespeed(rhoL,rhoUnL,EL)
        cR   = @. wavespeed(rhoR,rhoUnR,ER)
        lam  = @. max(abs(cL),abs(cR))
        return (@. lam*(uL-uR))
end
dLF(uL,uR,args...) = ForwardDiff.jacobian(uR->LF(uL,uR,args...),uR)

## mappings between conservative and entropy variables and vice vera
# dVdU_fun(U) = ForwardDiff.jacobian(U->SVector(v_ufun(U...)...),U)
# dUdV_fun(V) = ForwardDiff.jacobian(V->SVector(u_vfun(V...)...),V)
dVdU_fun(U) = dVdU_explicit(U)
dUdV_fun(V) = dUdV_explicit(V)

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
        Vq_fld  = droptol!(kron(Id_fld,Vq),1e-12)
        VhP     = Vh*Pq
        VhP_fld = droptol!(kron(Id_fld,VhP),1e-12)
        Vh_fld  = droptol!(kron(Id_fld,Vh),1e-12)
        M_fld   = droptol!(kron(Id_fld,M),1e-12)

        # init jacobian matrix (no need for entropy projection since we'll zero it out later)
        dFdU_h = repeat(I+Ax+Ay,Nfields,Nfields)

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

                fill!(dFdU_h.nzval,0.0)
                accum_hadamard_jacobian!(dFdU_h, Ax, dFx, Qh)
                accum_hadamard_jacobian!(dFdU_h, Ay, dFy, Qh)
                accum_hadamard_jacobian!(dFdU_h, B,  dLF, Qh, nxh,nyh) # flux term involving normals
                dVdU_h = banded_matrix_function(dVdU_fun, Uq)
                dUdV_h = banded_matrix_function(dUdV_fun, VUh)
                dFdU   = droptol!(transpose(Vh_fld)*(dFdU_h*dUdV_h*VhP_fld*dVdU_h*Vq_fld),1e-12)

                # solve and update
                dQ   = (M_fld + .5*dt*dFdU)\(M_fld*res)
                dQnorm = norm(dQ)/sum(norm.(Qprev))
                Qtmp = reshape(vcat(Qnew...) - dQ, length(Q[1]), Nfields)   # convert Qnew to column vector for update

                # return columnize(reshape(Qtmp,length(Q[1]),Nfields)),dQnorm # convert back to array of arrays
                for fld = 1:length(Qnew)
                        Qnew[fld] .= Qtmp[:,fld]
                end
                return dQnorm
        end
        return midpt_newton_iter!
end

## init condition, rhs

@unpack xq,yq = md
rho = vec(rd.Pq*(@. 1 + .1*(abs(xq)<.5).*(abs(yq)<.5)))
# rho = @. 2 + exp(-10*(x^2+y^2)) #ones(size(x))
rhou = zeros(size(x))
rhov = zeros(size(x))
E = @. rho^1.4
Q = [rho,rhou,rhov,E]
Qnew = copy.(Q)

# convert to tuple
Q = tuple(Q...)
Qnew = tuple(Qnew...)

# set time-stepping constants
CN = (N+1)*(N+2)/2  # estimated trace constant
h = minimum(md.J[1,:]./md.sJ[1,:]) # ratio of J/Jf = O(h^d/h^d-1)
dt = CFL * 2 * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# initialize jacobian
ops = (Ax,Ay,copy(transpose(Ax)),copy(transpose(Ay)),Bx,B,M,Vh,Ph,Vq,Pq) # pack inputs together
funs = (Fx,Fy,dFx,dFy,LF,dLF)
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
        while dQnorm > 1e-12
                # Qnew,dQnorm = midpt_newton_iter!(Qnew,Q)
                dQnorm = midpt_newton_iter!(Qnew,Q)
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
                println("Number of time steps $i out of $Nsteps")
                # display(scatter(x,Q[1]))
        end
end

@unpack VDM = rd
rp, sp = Basis2DTri.equi_nodes_2D(25)
Vp = Basis2DTri.vandermonde_2D(N,rp,sp)/VDM

gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=2)
xp,yp,up = (x->Vp*reshape(x,size(Vp,2),K)).((x,y,Q[1]))
# display(scatter(xp,yp,up,zcolor=up,cam=(3,25),axis=false))
scatter(xp,yp,up,zcolor=up,cam=(0,90),border=:none,axis=false)
# png("sol_unif_mesh.png")
