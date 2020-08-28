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
using UniformTriMesh

using SetupDG
using ExplicitJacobians

"Approximation parameters"
N = 2 # The order of approximation
K1D = 8
CFL = 1
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

println("Done initializing mesh + ref elements")

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
        Ph   = kron(spdiagm(0 => 1 ./ J[1,:]), sparse(Ph))
        x,y = (a->a[:]).((x,y))

        return Ax,Ay,AxTr,AyTr,Bx,B,VhTr,Vh,M,Ph,x,y
end

Ax,Ay,AxTr,AyTr,Bx,B,VhTr,Vh,M,Ph,x,y =
        build_global_ops(rd::RefElemData,md::MeshData)
println("Done building global ops")

# jacspy = I+Ax+Ay
# jac = droptol!(jacspy,1e-12)
# jac = droptol!(VhTr*jacspy*Vh,1e-12)
# display(spy(jac .!= 0,ms=2.25))
# # title="nnz = $(nnz(jac))",
# error("d")

## define Burgers fluxes
function F(uL,uR)
        Fx = @. (uL^2 + uL*uR + uR^2)/6
        Fy = @. 0*uL
        return Fx,Fy
end

# extract coordinate fluxes
Fx = (uL,uR)->F(uL,uR)[1]
Fy = (uL,uR)->F(uL,uR)[2]

# AD for jacobians
dFx(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[1],uR)
dFy(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[2],uR)

## nonlinear solver stuff
function init_newton_fxn(Q,ops,rd::RefElemData,md::MeshData,funs,dt)

        Ax,Ay,AxTr,AyTr,Bx,B,M,Vh,Ph = ops
        Fx,Fy,dFx,dFy,LF,dLF = funs

        # set up normals for use in penalty term
        @unpack Vq,Vf = rd
        @unpack nxJ,nyJ,sJ = md
        Nq,Np = size(Vq)
        Nf    = size(Vf,1)
        Nh    = Nq + Nf
        fids  = (Nq+1):Nh
        nxhmat,nyhmat = ntuple(x->zeros(Nh,K),2)
        nxhmat[fids,:] = nxJ[:]./sJ[:]
        nyhmat[fids,:] = nyJ[:]./sJ[:]
        nxh,nyh = (x->x[:]).((nxhmat,nyhmat))

        # get lengths of arrays
        Nfields = length(Q)
        Id_fields = speye(Nfields) # for Kronecker expansion to large matrices - fix later with lazy evals
        Vh_fields = droptol!(kron(Id_fields,Vh),1e-12)
        M_fields  = droptol!(kron(Id_fields,M),1e-12)

        # init jacobian matrix
        dFdU_h = repeat(I+Ax+Ay,Nfields,Nfields)

        function midpt_newton_iter!(Qnew, Qprev) # for Burgers' eqn specifically

                Qh    = SVector((x->Vh*x).(Qnew)) # tuples are faster, but need arrays for ForwardDiff

                ftmp  = hadamard_sum(AxTr,Fx,Qh) + hadamard_sum(AyTr,Fy,Qh) + hadamard_sum(B,LF,Qh,nxh,nyh)
                f     = vcat((x->Ph*x).(ftmp)...)
                res   = vcat(Qnew...) + .5*dt*f - vcat(Qprev...)

                fill!(dFdU_h.nzval,0.0)
                accum_hadamard_jacobian!(dFdU_h, Ax, dFx, Qh)
                accum_hadamard_jacobian!(dFdU_h, Ay, dFy, Qh)
                accum_hadamard_jacobian!(dFdU_h, B, dLF, Qh,nxh,nyh) # flux term involving normals
                dFdU = droptol!(transpose(Vh_fields)*(dFdU_h*Vh_fields),1e-12)

                # solve and update
                dQ   = (M_fields + .5*dt*dFdU)\(M_fields*res)
                Qtmp = reshape(vcat(Qnew...) - dQ,length(Q[1]),Nfields)       # convert Qnew to column vector for update
                for fld = 1:Nfields
                        Qnew[fld] .= Qtmp[:,fld]
                end
                return norm(dQ) # convert back to array of arrays
        end
        return midpt_newton_iter!
end

## init condition, rhs

u = @. -sin(pi*x)
# u = randn(size(x))
Q = [u]
Qnew = deepcopy(Q)

# convert to tuple
Q = tuple(Q...)
Qnew = tuple(Qnew...)

# set time-stepping constants
CN = (N+1)*(N+2)/2  # estimated trace constant
h = minimum(md.J[1,:]./md.sJ[1,:]) # ratio of J/Jf = O(h^d/h^d-1)
dt = CFL * 2 * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

function LF(uL,uR,nxL,nyL,nxR,nyR)
        absnx = @. (abs(nxL) + abs(nxR))/2
        return (@. max(abs(uL),abs(uR))*(uL-uR)*absnx)
end
dLF(uL,uR,args...) = ForwardDiff.jacobian(uR->LF(uL,uR,args...),uR)

# initialize Newton vars
ops = (Ax,Ay,copy(transpose(Ax)),copy(transpose(Ay)),Bx,B,M,Vh,Ph) # pack inputs together
fxns = (Fx,Fy,dFx,dFy,LF,dLF)
midpt_newton_iter! = init_newton_fxn(Q,ops,rd,md,fxns,dt)

## newton time iteration

bcopy!(utarget,u) = utarget .= u

it_count = zeros(Nsteps)
energy   = zeros(Nsteps)
for i = 1:Nsteps
        global Q,Qnew

        #Qnew = copy(Q)  # copy / over-write at each timestep
        bcopy!.(Qnew,Q)
        iter = 0
        dQnorm = 1
        while dQnorm > 1e-12
                dQnorm = midpt_newton_iter!(Qnew,Q)
                iter += 1
                if iter > 10
                        println("iter = $iter")
                end
        end
        it_count[i] = iter
        bcopy!.(Q, (@. 2*Qnew - Q)) # implicit midpoint rule

        u = Q[1]
        energy[i] = u'*M*u

        if i%10==0 || i==Nsteps
                println("Number of time steps $i out of $Nsteps")
                # display(scatter(x,Q[1]))
        end
end

@unpack VDM = rd
rp, sp = equi_nodes_2D(25)
Vp = vandermonde_2D(N,rp,sp)/VDM

xp,yp,up = (x->Vp*reshape(x,size(Vp,2),K)).((x,y,Q[1]))
gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=2)
# display(scatter(xp,yp,up,zcolor=up,cam=(3,25),axis=false))
scatter(xp,yp,up,zcolor=up,cam=(0,90),border=:none,axis=false)
# contourf(xp,yp,up,lw=1,fill=true,levels=20)
# png("sol_unif_mesh.png")
