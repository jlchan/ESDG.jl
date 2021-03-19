using NodesAndModes
using StartUpDG
using UnPack
using LinearAlgebra
using SparseArrays
using FluxDiffUtils
using StaticArrays
using Plots

N = 3
K = 16
T = 2.0
CFL = .125

interval = 10

rd = RefElemData(Line(),N; quad_rule_vol=gauss_lobatto_quad(0,0,N))
VX,EToV = uniform_mesh(Line(),K)
md_BCs = MeshData(VX,EToV,rd)
md = make_periodic(md_BCs,rd)

@unpack M,Dr,Vf,LIFT = rd

E = M*LIFT

QN = M*Dr
Q1 = diagm(1=>ones(N))-diagm(-1=>ones(N))
Q1[1,1] = -1
Q1[N+1,N+1] = 1
@. Q1 = .5*Q1

# Q1 = QN

# try FEM mass
# M = diagm(Q1*rd.r)
# M = I(N+1) * 2/(N+1)

ΔQ = (QN-QN')-(Q1-Q1')
S1 = (Q1-Q1')

D = Matrix(I(N+1))
D[end,end] = 0.0
F = rd.VDM*D/rd.VDM

@inline avg(x,y) = .5*(x+y)
function fS(UL,UR)
    hL,huL = UL
    hR,huR = UR
    uL,uR = huL/hL, huR/hR
    f1 = avg(huL,huR)
    f2 = avg(huL,huR)*avg(uL,uR) + .5*hL*hR
    return SVector{2}(f1,f2)
end
# function fS(UL,UR)
#     hL,huL = UL
#     hR,huR = UR
#     uL,uR = huL/hL, huR/hR
#     f1 = avg(huL,huR)
#     f2 = avg(huL*uL,huR*uR) + .5*hL*hR
#     return SVector{2}(f1,f2)
# end
function dS(UL,UR)
    hL,huL = UL
    hR,huR = UR
    uL,uR = huL/hL, huR/hR
    λL = abs(uL) + sqrt(abs(hL))
    λR = abs(uR) + sqrt(abs(hR))
    λ = .5*max(λL,λR)
    return SVector{2}(λ*(hL-hR),λ*(huL-huR))
end
# fS(uL,uR) = @. (uL*uL + uL*uR + uR*uR)/6.0
# dS(uL,uR) = @. .5*max(abs(uL),abs(uR))*(uL-uR)
# fLF(uL,uR) = .5*(uL^2/2 + uR^2/2)

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
Qg = build_rhs_matrix(u->rhs(u,(QN-QN'),2*E,rd,md),N+1,K)
Bg = build_rhs_matrix(u->rhsf(u,2*E,rd,md),N+1,K)
QgTr = -Qg # assume skew-symmetric
Q1g = build_rhs_matrix(u->rhs(u,(Q1-Q1'),2*E,rd,md),N+1,K)
Q1gTr = -Q1g # assume skew-symmetric

function hadsum(ATr::SparseMatrixCSC,F::Fxn,Q::SVector{N,T}) where {Fxn,N,T}
    rhs = similar.(Q)
    rows = rowvals(ATr)
    vals = nonzeros(ATr)
    for i = 1:size(ATr,2)
        Qi = getindex.(Q,i)
        val_i = MVector{N}(zero.(Qi))
        for row_id in nzrange(ATr,i)
            j = rows[row_id]
            val_i .+= vals[row_id].*F(Qi,getindex.(Q,j))
        end
        setindex!.(rhs,val_i,i)
    end
    return rhs
end

@unpack x,J = md
MJ = M*J
invm = 1 ./ vec(MJ)
x = vec(x)
# u = @. (x > -1/3)*1.0

h = @. 1e-1 + exp(-25*x^2)
# h = @. 1.0 + .5*(x > 0.0)
# b = @. .5*sin(pi*x)
b = @. 0*x
# bmax = maximum(b)
# h = @. 2.0 - b
hu = @. 0*x
Q = SVector{2}(h,hu)

dt0 = CFL*(x[2]-x[1]) / maximum(@. sqrt(abs(h)))
m_min = minimum(MJ)

Nsteps = ceil(Int,T/dt0)
dt = T/Nsteps
ubounds = (0.0,maximum(h))
l = zeros(N+1,md.K)

ops = Qg,QgTr,Q1g,Q1gTr,ΔQ,S1,invm
Qtmp1 = similar.(Q)
Qtmp2 = similar.(Q)
function update_FE!(Qnew,l,Q,ops,dt,b,N,K)

    Qg,QgTr,Q1g,Q1gTr,ΔQ,S1,invm = ops

    # low order update
    r1 = hadsum(Q1gTr,fS,Q) .+ hadsum(abs.(Q1gTr),dS,Q)
    r1[2] .+= .5*Q[1].*(Q1g*b)
    # r1[2] .+= .5*Q[1].*(Qg*b)
    ΔbN = .5*Q[1].*((Qg-Q1g)*b)
    Q1 = map((x,y)->x - dt.*invm.*y,Q,r1)

    # local high order updates
    val_e = SVector{2}([zeros(size(ΔQ,1)), zeros(size(ΔQ,1))]) # tmp storage
    for e = 1:K
        l_e = 1.0
        for i = 1:size(ΔQ,1)
            idi = i + (e-1)*(N+1)
            Qi = getindex.(Q,idi)
            val_i = MVector{2}(0.0,0.0)
            for j = 1:size(ΔQ,2)
                idj = j + (e-1)*(N+1)
                Qj = getindex.(Q,idj)
                Aij = ΔQ[i,j] .* fS(Qi,Qj) - abs(S1[i,j]) .* dS(Qi,Qj)

                λj = 1/(size(ΔQ,2)-1) # dim(N(i)\i)
                l_ij = min(1.0,abs(Qi[1]*λj/(dt*invm[idi]*Aij[1])))
                l_e = min(l_e,l_ij)

                val_i .+= Aij
                val_i[2] += ΔbN[idi]/size(ΔQ,2)
            end
            # val_i[2] += ΔbN[idi]
            setindex!.(val_e,val_i,i)
        end

        # l_e = 0.0
        l[:,e] .= l_e # store for plotting

        for i = 1:size(ΔQ,1)
            idi = i + (e-1)*(N+1)
            Qnew_i = getindex.(Q1,idi) - l_e.*dt.*invm[idi].*getindex.(val_e,i)
            setindex!.(Qnew,Qnew_i,idi) # local update - squeeze lij into here
        end
    end
    return Qnew,l
end

init_mass = sum(vec(MJ).*Q[1])
@gif for i = 1:Nsteps

    update_FE!(Qtmp1,l,Q,ops,dt,b,N,K)
    update_FE!(Qtmp2,l,Qtmp1,ops,dt,b,N,K)
    map((x,y)->y .= .5 .* (x .+ y),Qtmp2,Q)

    if i%interval==0
        println("$i / $Nsteps: minh = $(minimum(Q[1])), Δmass = $(sum(vec(MJ).*Q[1]) - init_mass)")
        plot(rd.Vp*reshape(x,N+1,K),rd.Vp*reshape(Q[1],N+1,K))
        scatter!(.5*rd.wq'*reshape(x,N+1,K),.5*rd.wq'*reshape(Q[1],N+1,K))
        # plot!(rd.Vp*reshape(x,N+1,K),rd.Vp*reshape(Q[2],N+1,K),ls=:dash)
        # scatter!(x,Q[1],ms=1,ylims = (-1.0,2.0))
        plot!(reshape(x,N+1,K),l,leg=false,ylims = (-1.0,2.0))
    end
end every interval

# Δmass =
# @show Δmass

# println("Δmass = $Δmass")

# plot(x,Q[1],mark=:dot,ms=2,ylims = ubounds .+ (-.5,.5))
# plot!(reshape(x,N+1,K),l,leg=false)
