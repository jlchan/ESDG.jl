using NodesAndModes
using StartUpDG
using UnPack
using LinearAlgebra
using SparseArrays
using FluxDiffUtils
using StaticArrays
using Plots

N = 5
K = 16
T = .1
CFL = .1

interval = 100

rd = RefElemData(Line(),N; quad_rule_vol=gauss_lobatto_quad(0,0,N))
VX,EToV = uniform_mesh(Line(),K)
md_BCs = MeshData(VX,EToV,rd)
md = make_periodic(md_BCs,rd)

@unpack M,Dr,Vf,LIFT = rd

Q = M*Dr
Q1 = diagm(1=>ones(N))-diagm(-1=>ones(N))
Q1[1,1] = -1
Q1[N+1,N+1] = 1
@. Q1 = .5*Q1

# Q1 = .95*Q1 + .05*Q
# Q1 = Q

E = M*LIFT

fS(uL,uR) = @. (uL*uL + uL*uR + uR*uR)/6.0
dS(uL,uR) = @. .5*max(abs(uL),abs(uR))*(uL-uR)
fLF(uL,uR) = .5*(uL^2/2 + uR^2/2)

function rhs(u,Qskew,E,rd,md)
    @unpack Vf = rd
    @unpack rxJ,nxJ,mapP = md
    uf = Vf*u
    return rxJ.*(Qskew*u) + .5*E*(uf[mapP].*nxJ)
end
function rhsf(u,E,rd,md)
    @unpack Vf = rd
    @unpack nxJ,mapP = md
    uf = Vf*u
    return .5*E*(uf[mapP].*nxJ)
end
function build_rhs_matrix(rhs::F,Np,K) where {F}
    A = zeros(Np*K,Np*K)
    u = zeros(Float64,Np,K)
    for i = 1:Np*K
        u[i] = 1.0
        A[:,i] .= vec(rhs(u))
        u[i] = 0.0
    end
    return droptol!(sparse(A),1e-12)
end
Qg = build_rhs_matrix(u->rhs(u,.5*(Q-Q'),E,rd,md),N+1,K)
Bg = build_rhs_matrix(u->rhsf(u,E,rd,md),N+1,K)
QgTr = -Qg # assume skew-symmetric
Q1g = build_rhs_matrix(u->rhs(u,.5*(Q1-Q1'),E,rd,md),N+1,K)
Q1gTr = -Q1g # assume skew-symmetric

# matrix used in convex limiting
ΔQ = .5*(Q1-Q1') - .5*(Q-Q')

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

@unpack x,J = md
MJ = M*J
invm = 1 ./ vec(M*J)
x = vec(x)
u = @. (x > -1/3)*1.0
u = @. exp(-25*x^2)
# u = @. 1-sin(pi*x)

u1 = similar(u)
uN = similar(u)
rhsu = similar(u)
l = reshape(similar(u),N+1,K)
L = zeros(N+1,N+1)

dt = CFL*(x[2]-x[1]) / maximum(abs.(u))
Nsteps = ceil(Int,T/dt)
dt = T/Nsteps
ubounds = extrema(u)

init_mass = sum(vec(MJ).*u)
@gif for i = 1:Nsteps
    global u
    r1 = hadsum(Q1gTr,fS,u) + .5*hadsum(abs.(Q1gTr),dS,u)
    rN = hadsum(QgTr,fS,u) + .5*hadsum(abs.(Bg),dS,u)

    # low order update
    @. u1 = u - dt*invm*r1
    @. uN = u - dt*invm*rN

    u = reshape(u,N+1,K)

    # local limiting
    for e = 1:K
        u1e = @view reshape(u1,N+1,K)[:,e]
        uNe = @view reshape(uN,N+1,K)[:,e]
        uNmin = minimum(uNe)
        θ = max(0.0,min(1.0, minimum(@. u1e / (u1e-uNmin + 1e-7))))
        θ = 0.0
        u[:,e] = @. u1e + θ*(uNe-u1e)
        l[:,e] .= 0.0
        if θ > 1e-12 # for plotting
            l[:,e] .= 1-θ
        end
    end

    # # node-wise convex limiting
    # for e = 1:K
    #     uL,uR = NodesAndModes.meshgrid(@view u[:,e])
    #     A = ΔQ .* fS(uL,uR) - .5*abs.(.5*(Q1-Q1')) .* dS(uL,uR)
    #     λij = 1/(N+1)
    #     for i = 1:N+1
    #         for j = 1:N+1
    #             λij = MJ[i,e]/(N+1)
    #             L[i,j] = 1.0
    #             if u[i,e] + dt*A[i,j]/λij > 0 # need u[i,e] + lij * dt * Aij / λij > 0
    #                 L[i,j] = 1.0
    #             elseif abs(A[i,j]) > 1e-14
    #                 L[i,j] = λij * (1e-16 - u[i,e]) / A[i,j]
    #             end
    #         end
    #     end
    #     # fill!(L,minimum(L))
    #     u[:,e] = reshape(u1,N+1,K)[:,e] + dt*sum(min.(L,L').*A,dims=2) ./ @view MJ[:,e]
    #     l[:,e] = 1 .- sum(min.(L,L'),dims=2)/(N+1)
    # end

    u = vec(u)

    if i%interval==0
        println("$i / $Nsteps: $ubounds, $(extrema(u))")
        plot(x,u,mark=:dot,ms=2,ylims=ubounds .+ (-1,1))
        plot!(reshape(x,N+1,K),l,leg=false)
    end
end every interval

Δmass = sum(vec(MJ).*u)-init_mass
println("Δmass = $Δmass")
plot(x,u,mark=:dot,ms=2,ylims=ubounds .+ (-1,1))
