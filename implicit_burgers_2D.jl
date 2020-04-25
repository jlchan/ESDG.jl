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
using BlockSparseMatrices

"Approximation parameters"
N = 2 # The order of approximation
K1D = 16
CFL = 100
T = 1 # endtime

"Mesh related variables"
VX, VY, EToV = uniform_tri_mesh(K1D)
# VX = @. VX - .3*sin(pi*VX)

# initialize ref element and mesh
# specify lobatto nodes to automatically get DG-SEM mass lumping
rd = init_reference_tri(N)
md = init_mesh((VX,VY),EToV,rd)

# Make domain periodic
@unpack Nfaces,Vf = rd
@unpack xf,yf,FToF,K,mapM,mapP,mapB = md
LX,LY = (x->maximum(x)-minimum(x)).((VX,VY)) # find lengths of domain
mapPB = build_periodic_boundary_maps!(xf,yf,LX,LY,Nfaces*K,mapM,mapP,mapB,FToF)
mapP[mapB] = mapPB
# @pack! md = mapP,FToF

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
@unpack rxJ, sxJ, ryJ, syJ = md
rxJ, sxJ, ryJ, syJ = (x->Vh*x).((rxJ, sxJ, ryJ, syJ)) # interp to hybridized points
# @pack! md = rxJ, sxJ, ryJ, syJ

Ax,Ay,Bx,By = assemble_global_SBP_matrices_2D(rd,md,Qrhskew,Qshskew)

# add off-diagonal couplings
Ax += Bx
Ay += By

AxTr = sparse(transpose(Ax))
AyTr = sparse(transpose(Ay))
Bx   = abs.(Bx) # for penalization term

# globalize operators
@unpack J = md
VhTr = kron(speye(K),sparse(transpose(Vh)))
Vh   = kron(speye(K),sparse(Vh))
invM = kron(spdiagm(0 => 1 ./ J[1,:]),sparse(inv(M)))
M    = kron(spdiagm(0 => J[1,:]),sparse(M))
Ph   = kron(spdiagm(0 => 1 ./ J[1,:]),sparse(Ph))

println("Done building global ops")

function F(uL,uR)
        Fx = @. (uL^2 + uL*uR + uR^2)/6
        Fy = @. 0*uL
        return Fx,Fy
end
function LF(uL,uR)
        return (@. max(abs(uL),abs(uR))*(uL-uR))
end

# extract coordinate fluxes
Fx = (uL,uR)->F(uL,uR)[1]
Fy = (uL,uR)->F(uL,uR)[2]

@unpack x,y = md
x,y = (a->a[:]).((x,y))

# set initial condition
u = @. -sin(pi*x)
Q = [u]

# set time-stepping constants
CN = (N+1)*(N+2)/2  # estimated trace constant
h = minimum(J)
dt = CFL * 2 * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

function midpt_newton_iter!(dFdU_h::SparseMatrixCSC, Qnew, Qprev, dF, ops, varargs...) # for Burgers' eqn specifically

        dFx,dFy,dLF = dF
        Ax,Ay,AxTr,AyTr,Bx,Vh = ops

        # get lengths of arrays
        Nfields = length(Q)
        Id_fields = speye(Nfields) # for Kronecker expansion to large matrices - fix later with lazy evals
        Vh_fields = droptol!(kron(Id_fields,Vh),1e-12)

        Qh    = (x->Vh*x).(SVector{Nfields}(Qnew)) # tuples are faster, but need SVector for ForwardDiff
        ftmp  = hadamard_sum(AxTr,Fx,Qh) + hadamard_sum(AyTr,Fy,Qh) + hadamard_sum(Bx,LF,Qh)
        f     = kron(Id_fields,Ph)*vcat(ftmp...)
        res   = vcat(Qnew...) + .5*dt*f - vcat(Qprev...)

        # #dFdU_h = hadamard_jacobian(Ax, dFx, Qh) + hadamard_jacobian(Ay, dFy, Qh) + hadamard_jacobian(Bx,dLF,Qh)
        fill!(dFdU_h.nzval,0.0)
        accum_hadamard_jacobian!(dFdU_h, Ax, dFx, Qh)
        accum_hadamard_jacobian!(dFdU_h, Ay, dFy, Qh)
        accum_hadamard_jacobian!(dFdU_h, Bx, dLF, Qh)
        dFdU   = droptol!(transpose(Vh_fields)*(dFdU_h*Vh_fields),1e-12)

        dQ   = (kron(Id_fields,M) + .5*dt*dFdU)\(kron(Id_fields,M)*res)
        Qnew = vcat(Qnew...) - dQ  # convert to global column, can also scale by M for additional sparsity
        Qnew = columnize(reshape(Qnew,length(Q[1]),Nfields)) # convert back to array of arrays

        return Qnew,norm(dQ)
end

dFx(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[1],uR)
dFy(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[2],uR)
dLF(uL,uR) = ForwardDiff.jacobian(uR->LF(uL,uR),uR)

# pack inputs together
dF = (dFx,dFy,dLF)
ops = (Ax,Ay,copy(transpose(Ax)),copy(transpose(Ay)),Bx,Vh)

# initialize jacobian
Qh = (x->Vh*x).(SVector{length(Q)}(Q))
dFdU_h = hadamard_jacobian(Ax, dFx, Qh) + hadamard_jacobian(Ay, dFy, Qh) + hadamard_jacobian(Bx,dLF,Qh)

## switch to block ops

# Nh,Np = size([Vq;Vf])
# AxBlk = SparseMatrixBSC(Ax,Nh,Nh)
# AyBlk = SparseMatrixBSC(Ay,Nh,Nh)
# BxBlk = SparseMatrixBSC(Bx,Nh,Nh)
# block_ops = (AxBlk,AyBlk,BxBlk,[Vq;Vf])

# function init_DG_jacobian(Nh,md::MeshData)
#         @unpack FToF = md
#         EToE = @. (FToF.-1) ÷ Nfaces + 1
#         nzids = findall(EToE .!= 0)
#         rows = (x->x[2]).(nzids)
#         cols = EToE[nzids]
#
#         #append rows/cols for diagonal blocks
#         K = size(EToE,2)
#         rows = append!(rows,1:K)
#         cols = append!(cols,1:K)
#         return block_spzeros(Nh,Nh,rows,cols)
# end

# function midpt_newton_iter!(dFdU_h::Array{SparseMatrixBSC{Tv,Ti},Td}, Qnew, Qprev,
#                             dF, ops, block_ops) where {Tv,Ti,Td}
#
#         dFx,dFy,dLF = dF
#         Ax,Ay,AxTr,AyTr,Bx,Vh = ops
#         AxBlk,AyBlk,BxBlk,Vh_local = block_ops
#
#         # get lengths of arrays
#         Nfields = length(Q)
#         Id_fields = speye(Nfields) # for Kronecker expansion to large matrices - fix later with lazy evals
#         Vh_fields = droptol!(kron(Id_fields,Vh),1e-12)
#
#         Qh    = (x->Vh*x).(SVector{Nfields}(Qnew)) # tuples are faster, but need SVector for ForwardDiff
#         ftmp  = hadamard_sum(AxTr,Fx,Qh) + hadamard_sum(AyTr,Fy,Qh) + hadamard_sum(Bx,LF,Qh)
#         f     = kron(Id_fields,Ph)*vcat(ftmp...)
#         res   = vcat(Qnew...) + .5*dt*f - vcat(Qprev...)
#
#         # #dFdU_h = hadamard_jacobian(Ax, dFx, Qh) + hadamard_jacobian(Ay, dFy, Qh) + hadamard_jacobian(Bx,dLF,Qh)
#         fill!.(dFdU_h,0.0)
#         accum_hadamard_jacobian!(dFdU_h, AxBlk, dFx, Qh)
#         accum_hadamard_jacobian!(dFdU_h, AyBlk, dFy, Qh)
#         accum_hadamard_jacobian!(dFdU_h, BxBlk, dLF, Qh)
#         dFdU   = block_lrmul(dFdU_h[1],transpose(Vh_local),Vh_local)
#         dFdU   = SparseMatrixCSC(dFdU) # generalize beyond scalar?
#
#         b    = (kron(Id_fields,M)*res)
#         dQ   = (kron(Id_fields,M) + .5*dt*dFdU)\b
#         Qnew = vcat(Qnew...) - dQ  # convert to global column, can also scale by M for additional sparsity
#         Qnew = columnize(reshape(Qnew,length(Q[1]),Nfields)) # convert back to array of arrays
#
#         return Qnew,norm(dQ)
# end

# dFdU_h = [init_DG_jacobian(Nh,md)]
# accum_hadamard_jacobian!(dFdU_h, AxBlk, dFx, Qh)
# accum_hadamard_jacobian!(dFdU_h, AyBlk, dFy, Qh)
# accum_hadamard_jacobian!(dFdU_h, BxBlk, dLF, Qh)
#
# dFdU_h_CSC = SparseMatrixCSC(dFdU_h[1])
# @btime accum_hadamard_jacobian!($dFdU_h_CSC, $Ax, $dFx, $Qh)
# @btime accum_hadamard_jacobian!($dFdU_h, $AxBlk, $dFx, $Qh)
# error("d")

##

it_count = zeros(Nsteps)
for i = 1:Nsteps
        global Q #,dFdU_h

        Qnew = copy(Q)  # copy / over-write
        iter = 0
        dQnorm = 1
        while dQnorm > 1e-12

                Qnew,dQnorm = midpt_newton_iter!(dFdU_h,Qnew,Q,dF,ops)

                iter += 1
                if iter > 10
                        println("iter = $iter")
                end
        end
        Q = @. 2*Qnew-Q # implicit midpoint rule
        it_count[i] = iter

        if i%10==0 || i==Nsteps
                println("Number of time steps $i out of $Nsteps")
                # display(scatter(x,Q[1]))
        end
end

@unpack Vp = rd
gr(aspect_ratio=1, legend=false,
markerstrokewidth=0, markersize=2)
xp,yp,vv = (x->Vp*reshape(x,size(Vp,2),K)).((x,y,Q[1]))
display(scatter(xp,yp,vv,zcolor=vv,cam=(3,25)))
# scatter(xp,yp,vv,zcolor=vv,cam=(0,90))
