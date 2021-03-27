using NodesAndModes
using StartUpDG
using UnPack
using LinearAlgebra
using SparseArrays
using StaticArrays
using Plots
using Test
using ForwardDiff

N = 2
K = 10
T = 20.0
CFL = .1

u0min = 1

interval = 500

vol_quad = gauss_lobatto_quad(0,0,N)
vol_quad = gauss_quad(0,0,2*N+1)
rd = RefElemData(Line(),N; quad_rule_vol=vol_quad)

VX,EToV = uniform_mesh(Line(),K)
md_BCs = MeshData(VX,EToV,rd)
md = make_periodic(md_BCs,rd)

function build_sbp(rd)
    @unpack M,Dr,Vq,Vf,Pq = rd
    Q = Pq'*(M*Dr)*Pq
    B = diagm([-1.,1.])
    E = Vf*Pq
    Qh = .5*[Q-Q' E'*B
            -B*E 0*B]
    Vh = [Vq;Vf]
    Ph = M\Vh'
    return Qh,E,Vh,Ph
end
Qh,E,Vh,Ph = build_sbp(rd)
ops = (Qh,E)
Nf,Nq = size(E)

@unpack M,Dr,Vq,Vf,Pq = rd

# Q = Pq'*(M*Dr)*Pq
# E = zeros(2,Nq)
# E[[1;end]] .= 1
# Qh = .5*[Q-Q' E'*B
#         -B*E 0*B]
# ops = (Qh,E)

# rd1 = RefElemData(Line(),1; quad_rule_vol=gauss_lobatto_quad(0,0,N))
# Qh1,E1,Vh1,Ph1 = build_sbp(rd1)
# E1 = zeros(2,N+1)
# E1[[1;end]] .= 1
# Vh = [I(N+1);E1]
# Q1 = Vh'*(Qh1 + .5*diagm([zeros(size(Qh1,2)-2);-1.;1.]))*Vh
# Qh1 = .5*[Q1-Q1' E1'*B
#         -B*E1 0*B]
# M = diagm(rd.wq)
# # M = inv(rd.VDM*rd.VDM')
# Ph = M\Matrix(Vh')
# ops = (Qh1,E1)

function rhs(uh,ops,rd,md)
    @unpack rxJ,nxJ,mapP = md
    Qh,E = ops
    Nf,Nq = size(E)
    fids = Nq+1:(Nq+Nf)
    uf = uh[fids,:]
    rhs = rxJ[1].*(Qh*uh)
    @. rhs[fids,:] += .5*uf[mapP]*nxJ
    return rhs
end
function rhsf(uh,ops,rd,md)
    @unpack rxJ,nxJ,mapP = md
    Qh,E = ops
    Nf,Nq = size(E)
    fids = Nq+1:(Nq+Nf)
    uf = uh[fids,:]
    rhs = zeros(size(uh))
    @. rhs[fids,:] = .5*uf[mapP]*nxJ
    return rhs
end
function build_rhs_matrix(rhs::F,Np,K) where {F}
    A = zeros(Np*K,Np*K)
    u = zeros(Float64,Np,K)
    for i = 1:Np*K
        u[i] = 1.0
        A[:,i] .= vec(rhs(u))
        u[i] = 0.0
    end
    return droptol!(sparse(A),1e-13)
end
Qg = droptol!(build_rhs_matrix(u->rhs(u,ops,rd,md),Nq+Nf,K),1e-12)
Bg = build_rhs_matrix(u->rhsf(u,ops,rd,md),Nq+Nf,K)
QgTr = -copy(Qg) # assume skew-symmetric (premult factor 2)

Vqg = kron(I(K),sparse(Vq))
Vhg = kron(I(K),sparse(Vh))
Phg = kron(diagm(vec(1 ./ md.J[1,:])),sparse(Ph))
Pqg = kron(I(K),sparse(Pq))
Mg = kron(spdiagm(0=>vec(md.J[1,:])),M)
invMg = kron(diagm(vec(1 ./ md.J[1,:])), inv(M))

function hadsum(ATr::SparseMatrixCSC,F::Fxn,u) where {Fxn}
    rhs = similar(u)
    rows = rowvals(ATr)
    vals = nonzeros(ATr)
    for i = 1:size(ATr,2)
        ui = u[i]
        val_i = zero(ui)
        for row_id in nzrange(ATr,i)
            j = rows[row_id]
            val_i += vals[row_id]*F(ui,u[j])
        end
        rhs[i] = val_i
    end
    return rhs
end

function hadjac(A,dF::Fxn,u) where {Fxn}
    jac = similar(A)
    rows = rowvals(A)
    vals = nonzeros(A)
    for j = 1:size(A,2)
        uj = u[j]
        val_j = zero(uj)
        for row_id in nzrange(A,j)
            i = rows[row_id]
            jac[i,j] = vals[row_id]*dF(u[i],uj)
        end
    end
    jac -= spdiagm(0=>vec(sum(jac,dims=1)))
    return jac
end

@unpack x,J = md
x = vec(x)

u0(x) = sin(pi*(x-.7)) + 2
u0(x) = u0min+rand()
u = @. u0(x) + 0e-3*cos(pi*x)
# u = u0.(x) + randn(size(x))

fEC(uL,uR) = @. (uL*uL + uL*uR + uR*uR)/6.0
# fEC(uL,uR) = @. (uL*uL + uR*uR)/4.0
dS(uL,uR) = @. 0*.5*max(abs(uL),abs(uR))*(uL-uR)
# fLF(uL,uR) = .5*(uL^2/2 + uR^2/2)
# fLF(uL,uR) = .5*(uL+uR)

# dF(uL,uR) = ForwardDiff.derivative(uR->fLF(uL,uR),uR)
dF(uL,uR) = ForwardDiff.derivative(uR->fEC(uL,uR),uR)
dD(uL,uR) = ForwardDiff.derivative(uR->dS(uL,uR),uR)

jac = Vhg'*(hadjac(Qg,dF,Vhg*u)-hadjac(abs.(Bg),dD,Vhg*u))*Vhg

# WJ = kron(diagm(J[1,:]),diagm([rd.wq; 0; 0]))
# scatter(eigvals(Matrix(hadjac(Qg,dF,Vhg*u)),WJ))

# # projection-based (dense SAT) Gauss
# QgRed = Pqg'*Vhg'*Qg*Vhg*Pqg
# jac = Vqg'*hadjac(QgRed,dF,Vqg*u)*Vqg

λ = eigvals(Matrix(jac),Matrix(Mg))
scatter(λ,title="Max real part = $(maximum(real(λ)))")


# b0 = Phg*(hadsum(QgTr,fLF,Vhg*u0.(x)) + .5*hadsum(abs.(Bg),dS,Vhg*u0.(x)))
# resu = zeros(size(u))
# rk4a,rk4b,rk4c = ck45()
#
# dt = CFL*(x[2]-x[1]) / maximum(abs.(u))
# Nsteps = ceil(Int,T/dt)
# dt = T/Nsteps
# ubounds = extrema(u)
# unorm = zeros(Nsteps)
# #@gif
# for i = 1:Nsteps
#     for INTRK = 1:5
#         uh = Vhg*u
#         rhsu = b0 - Phg*(hadsum(QgTr,fLF,uh) + .5*hadsum(abs.(Bg),dS,uh))
#         # rhsu = b0 - invM*hadsum(QgTrRed,fLF,u)
#         @. resu = rk4a[INTRK]*resu + dt*rhsu
#         @. u   += rk4b[INTRK]*resu
#     end
#
#     unorm[i] = sum(md.J[1]*rd.wq'*(rd.Vq*reshape(u,N+1,K)).^2)
#
#     if i%interval==0
#         println("$i / $Nsteps: $ubounds, $(extrema(u))")
#         #plot(x,u,mark=:dot,ms=2,ylims=ubounds .+ (-1,1))
#     end
# end #every interval
#
# plot((1:Nsteps)*dt,unorm,yaxis=:log)
# # plot(x, u, mark=:dot, ms=2, ylims=ubounds .+ (-1,1))
