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
K1D = 8
CFL = 100
T = 1 # endtime

"Mesh related variables"
VX, VY, EToV = uniform_tri_mesh(K1D)
VX = @. VX - .3*sin(pi*VX)

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
Qrhskew = .5*(Qrh-transpose(Qrh))
Qshskew = .5*(Qsh-transpose(Qsh))

# interpolate geofacs to both vol/surf nodes
@unpack rxJ, sxJ, ryJ, syJ = md
rxJ, sxJ, ryJ, syJ = (x->Vh*x).((rxJ, sxJ, ryJ, syJ)) # interp to hybridized points

## global matrices

Ax,Ay,Bx,By = assemble_global_SBP_matrices_2D(rd,md,Qrhskew,Qshskew)

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
invM = kron(spdiagm(0 => 1 ./ J[1,:]), sparse(inv(M)))
M    = kron(spdiagm(0 => J[1,:]),      sparse(M))
Ph   = kron(spdiagm(0 => 1 ./ J[1,:]), sparse(Ph))
x,y = (a->a[:]).((x,y))

println("Done building global ops")

## define Burgers fluxes
function F(uL,uR)
        Fx = @. (uL^2 + uL*uR + uR^2)/6
        Fy = @. 0*uL
        return Fx,Fy
end

function LF(uL,uR)
        return (@. max(abs(uL),abs(uR))*(uL-uR))
        # return @. (uL-uR)
end
function Ujump(uL,uR)
        return (@. (uL-uR))
end
function Lmax(uL,uR)
        return (@. max(abs(uL),abs(uR)))
end

# extract coordinate fluxes
Fx = (uL,uR)->F(uL,uR)[1]
Fy = (uL,uR)->F(uL,uR)[2]

# AD for jacobians
dFx(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[1],uR)
dFy(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[2],uR)
dLF(uL,uR) = ForwardDiff.jacobian(uR->LF(uL,uR),uR)
dUjump(uL,uR) = ForwardDiff.jacobian(uR->Ujump(uL,uR),uR)


## nonlinear solver stuff
function init_newton_fxn(Q,ops,rd::RefElemData,md::MeshData)

        Ax,Ay,AxTr,AyTr,Bx,Vh = ops

        # get lengths of arrays
        Nfields = length(Q)
        Id_fields = speye(Nfields) # for Kronecker expansion to large matrices - fix later with lazy evals
        Vh_fields = droptol!(kron(Id_fields,Vh),1e-12)
        M_fields  = droptol!(kron(Id_fields,M),1e-12)
        Ph_fields = droptol!(kron(Id_fields,Ph),1e-12)

        @unpack K = md
        @unpack Vq,Vf = rd
        Nq,Np = size(Vq)
        Nf,_ = size(Vf)
        Nh = Nq+Nf
        fids = Nq+1:Nh

        # init jacobian matrix
        Qh = (x->Vh*x).(SVector{length(Q)}(Q))
        BLFmat = hadamard_jacobian(Bx,dLF,Qh)
        dFdU_h = hadamard_jacobian(Ax, dFx, Qh) + hadamard_jacobian(Ay, dFy, Qh) + BLFmat

        function midpt_newton_iter!(Qnew, Qprev) # for Burgers' eqn specifically

                Qh    = (x->Vh*x).(SVector{Nfields}(Qnew)) # tuples are faster, but need SVector for ForwardDiff

                # frozen coefficient penalty
                fill!(BLFmat.nzval,0.0)
                hadamard_scale!(BLFmat,Bx,Lmax,Qh)

                # ftmp  = hadamard_sum(AxTr,Fx,Qh) + hadamard_sum(AyTr,Fy,Qh) + hadamard_sum(Bx,LF,Qh)
                function fres(Qh)
                        ftmp  = hadamard_sum(AxTr,Fx,Qh) + hadamard_sum(AyTr,Fy,Qh) + hadamard_sum(BLFmat,Ujump,Qh)
                        f     = Ph_fields*vcat(ftmp...)
                        res   = vcat(Qnew...) + .5*dt*f - vcat(Qprev...)
                        return res
                end
                res = fres(Qh)

                fill!(dFdU_h.nzval,0.0)
                accum_hadamard_jacobian!(dFdU_h, Ax, dFx, Qh)
                accum_hadamard_jacobian!(dFdU_h, Ay, dFy, Qh)
                accum_hadamard_jacobian!(dFdU_h, Bx, dLF, Qh)
                # accum_hadamard_jacobian!(dFdU_h, BLFmat, dUjump, Qh)

                dFdU   = droptol!(transpose(Vh_fields)*(dFdU_h*Vh_fields),1e-12)

                # TODO: add line search
                dQ   = (M_fields + .5*dt*dFdU)\(M_fields*res)
                # function linesearch(Qnew,dQ)
                #         iter = 0
                #         tau = .75
                #         resnorm = norm(res)
                #         resnew = copy(resnorm) + 1
                #         while (resnew > resnorm) & (iter < 5)
                #                 α = tau^iter
                #                 Qtmp = columnize(reshape(vcat(Qnew...) - α*dQ,length(Q[1]),Nfields))
                #                 resnew = norm(fres(Qtmp))
                #                 iter += 1
                #         end
                #         return tau^(iter-1)
                # end
                # α = linesearch(Qnew,dQ)
                # Qnew = vcat(Qnew...) - α*dQ                            # convert Qnew to column vector for update
                Qnew = vcat(Qnew...) - dQ                            # convert Qnew to column vector for update
                Qnew = columnize(reshape(Qnew,length(Q[1]),Nfields)) # convert back to array of arrays

                return Qnew,norm(dQ)
        end
        return midpt_newton_iter!
end

# pack inputs together
ops = (Ax,Ay,copy(transpose(Ax)),copy(transpose(Ay)),Bx,Vh)

## init condition, rhs

u = @. -sin(pi*x)
Q = [u]

# set time-stepping constants
CN = (N+1)*(N+2)/2  # estimated trace constant
h = minimum(J)
dt = CFL * 2 * h / CN
dt = .1
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

# initialize jacobian
midpt_newton_iter! = init_newton_fxn(Q,ops,rd,md)

## newton time iteration

it_count = zeros(Nsteps)
energy   = zeros(Nsteps)
for i = 1:Nsteps
        global Q

        Qnew = copy(Q)  # copy / over-write at each timestep
        iter = 0
        dQnorm = 1
        while dQnorm > 1e-12
                Qnew,dQnorm = midpt_newton_iter!(Qnew,Q)
                iter += 1
                if iter > 10
                        println("iter = $iter")
                end
        end
        it_count[i] = iter
        Q = @. 2*Qnew - Q # implicit midpoint rule

        u = Q[1]
        energy[i] = u'*M*u

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
scatter(xp,yp,vv,zcolor=vv,cam=(0,90))

# plot()
# for e = 1:K
#         vids = [EToV[e,:];EToV[e,1]]
#         vx = VX[vids]
#         vy = VY[vids]
#         plot!(vx,vy,linecolor=:black)
# end
# display(plot!(border=:none))
# png("squeezed_mesh")
