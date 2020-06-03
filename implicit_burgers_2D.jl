using Revise # reduce need for recompile
using Plots
using LinearAlgebra
using SparseArrays
using StaticArrays
using UnPack

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D
using SetupDG
using ExplicitJacobians

N = 2
K = 16
T = 1
CFL = 100

rd = init_reference_interval(N)

# Mesh related variables"
VX = LinRange(-1,1,K+1)
VX = @. VX - .32*sin(pi*VX)
EToV = transpose(reshape(sort([1:K; 2:K+1]),2,K))

@unpack V1 = rd
x = V1*VX[transpose(EToV)]
mapM = reshape(1:2*K,2,K)
mapP = copy(mapM)
mapP[1,2:end] .= mapM[2,1:end-1]
mapP[2,1:end-1] .= mapM[1,2:end]
mapP[1] = mapM[end]; mapP[end] = mapM[1] # Make periodic"

# Geometric factors and surface normals
J = repeat(transpose(diff(VX)/2),N+1,1)
nxJ = repeat([-1;1],1,K)
rxJ = 1

@unpack M,Dr,Vq,Vf,Pq = rd
Qr = Pq'*(M*Dr)*Pq
E  = Vf*Pq
Br = diagm([-1;1])
Qh = .5*[Qr-Qr' E'*Br;
        -Br*E 0*Br]
Vh = [Vq;Vf]
Ph = M\transpose(Vh)

M  = kron(spdiagm(0 => J[1,:]),M)
Vh = kron(speye(K),Vh)
Ph = kron(spdiagm(0 => 1 ./ J[1,:]),Ph)

function applyQh(u)
        Nf,Nq = size(E)
        uf = u[Nq+1:end,:]
        rhs = Qh*u
        rhs[Nq+1:end,:] += (@. .5*uf[mapP]*nxJ)
        return (@. 2*rhs)
end
function applyJump(u)
        Nf,Nq = size(E)
        uf = u[Nq+1:end,:]
        rhs = zeros(size(u))
        rhs[Nq+1:end,:] = (uf[mapP]-uf)
        return rhs
end

function build_rhs_matrix(applyRHS,Np,K)
        u = zeros(Np,K)
        A = zeros(Np*K,Np*K)
        for i in eachindex(u)
                u[i] = 1
                r_i = applyRHS(u)
                A[:,i] = r_i[:]
                u[i] = 0
        end
        return A
end
Nh = size(Qh,2)
A = build_rhs_matrix(applyQh,Nh,K)
B = build_rhs_matrix(applyJump,Nh,K)
A = droptol!(sparse(A),1e-12)
B = droptol!(sparse(B),1e-12)
# K = .5*A - .5*B
ATr = droptol!(sparse(transpose(A)),1e-12) # store transpose for col

x = x[:]
J = J[:]
F(uL,uR) = (@. (uL^2 + uL*uR + uR^2)/6)
# F(uL,uR) = (@. (uL + uR)/2)
LF(uL,uR) = (@. .5*max(abs(uL),abs(uR))*(uL-uR))

dF(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR),uR)
dLF(uL,uR) = ForwardDiff.jacobian(uR->LF(uL,uR),uR)


#use ATr for faster col access
function hadamard_sum_scalar(ATr,F,u)
        cols = rowvals(ATr)
        vals = nonzeros(ATr)
        m, n = size(ATr)
        rhs = zeros(n)
        for i = 1:n
                ui = u[i]
                val_i = 0.0
                for j in nzrange(ATr, i)
                        col = cols[j]
                        Aij = vals[j]
                        uj = u[col]
                        val_i += Aij*F(ui,uj)
                end
                rhs[i]= val_i
        end
        return rhs
end

CN = (N+1)*(N+2)/2  # estimated trace constant
h = minimum(J)
dt = CFL * 2 * h / CN
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

u = @. -sin(pi*x)
unew = copy(u)
energy = zeros(Nsteps)
it_count = zeros(Nsteps)
for i = 1:Nsteps
        global unew, u
        if false
                uh = Vh*u
                rhsu = -Ph*(hadamard_sum(ATr,F,uh) + hadamard_sum(B,LF,uh))
                u1 = u + dt*rhsu

                u1h = Vh*u1
                rhsu += -Ph*(hadamard_sum(ATr,F,u1h) + hadamard_sum(B,LF,u1h))
                @. u += .5*dt*rhsu

        else # implicit midpoint rule

                rnorm = 1
                iter = 0

                unew .= u  # copy
                uh    = Vh*unew
                f     = Ph*(hadamard_sum_scalar(ATr,F,uh) + hadamard_sum_scalar(abs.(B),LF,uh))
                res   = unew + .5*dt*f - u
                while rnorm > 1e-12

                        dFdU_h = hadamard_jacobian(A, dF, @SVector [uh]) + hadamard_jacobian(abs.(B), dLF, @SVector [uh])
                        dFdU   = Ph*dFdU_h*Vh
                        unew   = unew - (I + .5*dt*dFdU)\res # can also scale by M for additional sparsity

                        uh    = Vh*unew
                        f     = Ph*(hadamard_sum_scalar(ATr,F,uh) + hadamard_sum_scalar(abs.(B),LF,uh))
                        res   = unew + .5*dt*f - u
                        rnorm = norm(res)
                        iter += 1
                end
                it_count[i] = iter
                u = 2*unew-u # implicit midpoint rule
        end

        energy[i] = u'*(M*u)

        if i%10==0 || i==Nsteps
                println("Number of time steps $i out of $Nsteps")
        end
end

@unpack Vp = rd
Np = size(Vp,2)
xp,up = (x->Vp*reshape(x,Np,K)).((x,u))
display(scatter(xp,up))
