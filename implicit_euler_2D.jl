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
N = 2 # The order of approximation
K1D = 8
CFL = 10
T = .5 # endtime

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
Vq   = kron(speye(K),sparse(Vq))
Pq   = kron(speye(K),sparse(Pq))
VhTr = kron(speye(K),sparse(transpose(Vh)))
M    = kron(spdiagm(0 => J[1,:]),sparse(M))
Vh   = kron(speye(K),sparse(Vh))
Ph   = kron(spdiagm(0 => 1 ./ J[1,:]), sparse(Ph))
VhP  = Vh*Pq
x,y = (a->a[:]).((x,y))

println("Done building global ops")

## define Euler fluxes
function F(UL,UR)
    # convert to flux variables
    function UtoQ(U)
        rho,rhou,rhov,E = U
        beta = betafun(U...)
        return (rho,rhou./rho,rhov./rho,beta),(log.(rho),log.(beta))
    end
    QL,QlogL = UtoQ(UL)
    QR,QlogR = UtoQ(UR)
    Fx,Fy = euler_fluxes(QL...,QR...,QlogL...,QlogR...)
    return SVector{length(Fx)}(Fx...),SVector{length(Fy)}(Fy...)
end

# extract coordinate fluxes
Fx = (uL,uR)->F(uL,uR)[1]
Fy = (uL,uR)->F(uL,uR)[2]

# AD for jacobians
dFx(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[1],uR)
dFy(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR)[2],uR)

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

## transform mappings

dVdU_fun(U) = ForwardDiff.jacobian(U->SVector(v_ufun(U...)...),U)
dUdV_fun(V) = ForwardDiff.jacobian(V->SVector(u_vfun(V...)...),V)

## nonlinear solver stuff
function init_newton_fxn(Q,ops,rd::RefElemData,md::MeshData)

        # set up normals for use in penalty term
        @unpack Vq,Vf = rd
        @unpack nxJ,nyJ,sJ = md
        Nq,Np = size(Vq)
        Nf    = size(Vf,1)
        Nh    = Nq + Nf
        fids  = Nq + 1:Nh
        nxh,nyh = ntuple(x->zeros(Nh,K),2)
        nxh[fids,:] = @. nxJ/sJ
        nyh[fids,:] = @. nyJ/sJ
        nxh,nyh = vec.((nxh,nyh))

        # get lengths of arrays
        Ax,Ay,AxTr,AyTr,Bx,Vh,Vq,Pq = ops

        Nfields = length(Q)
        Id_fld = speye(Nfields) # for Kronecker expansion to large matrices - fix later with lazy evals
        Vq_fld = droptol!(kron(Id_fld,Vq),1e-12)
        VhP    = Vh*Pq
        VhP_fld = droptol!(kron(Id_fld,VhP),1e-12)
        Vh_fld = droptol!(kron(Id_fld,Vh),1e-12)
        M_fld  = droptol!(kron(Id_fld,M),1e-12)
        Ph_fld = droptol!(kron(Id_fld,Ph),1e-12)

        # init jacobian matrix (no need for entropy projection since we'll zero it out later)
        dFdU_h = repeat(I+Ax+Ay,Nfields,Nfields)

        function midpt_newton_iter!(Qnew, Qprev) # for Burgers' eqn specifically

                # perform entropy projection
                Uq    = (x->Vq*x).(SVector{Nfields}(Qnew))
                VUh   = SVector{Nfields}((x->VhP*x).(v_ufun(Uq...)))
                Qh    = SVector{Nfields}(u_vfun(VUh...))

                ftmp  = hadamard_sum(AxTr,Fx,Qh) + hadamard_sum(AyTr,Fy,Qh) + hadamard_sum(B,LF,Qh,nxh,nyh)
                f     = Ph_fld*vcat(ftmp...)
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
                # compute damping factor
                Qnew = vcat(Qnew...) - dQ                            # convert Qnew to column vector for update
                Qnew = columnize(reshape(Qnew,length(Q[1]),Nfields)) # convert back to array of arrays

                return Qnew,norm(dQ)
        end
        return midpt_newton_iter!
end

# pack inputs together
ops = (Ax,Ay,copy(transpose(Ax)),copy(transpose(Ay)),Bx,Vh,Vq,Pq)

## init condition, rhs

rho = @. 2 + (abs(x)<.5) #+ exp(-10*(x^2+y^2)) #ones(size(x))
rhou = zeros(size(x))
rhov = zeros(size(x))
E = rho.^1.4
Q = @SVector [rho,rhou,rhov,E]
# rho,u,v,p = vortex(x,y,0)
# Q = SVector{4}(primitive_to_conservative(rho,u,v,p))


# set time-stepping constants
CN = (N+1)*(N+2)/2  # estimated trace constant
h = minimum(J)
dt = CFL * 2 * h / CN
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
                if iter > 15
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

@unpack VDM = rd
rp, sp = Basis2DTri.equi_nodes_2D(25)
Vp = Basis2DTri.vandermonde_2D(N,rp,sp)/VDM

gr(aspect_ratio=1, legend=false,
   markerstrokewidth=0, markersize=2)
xp,yp,up = (x->Vp*reshape(x,size(Vp,2),K)).((x,y,Q[1]))
# display(scatter(xp,yp,up,zcolor=up,cam=(3,25),axis=false))

# plotly()
scatter(xp,yp,up,zcolor=up,cam=(0,90),border=:none,axis=false)
# png("sol_unif_mesh.png")
