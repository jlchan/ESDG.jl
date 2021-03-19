using NodesAndModes
using StartUpDG
using UnPack
using LinearAlgebra
using SparseArrays
using FluxDiffUtils
using StaticArrays
using Plots

N = 3
K = 64
T = 1.5
CFL = .125

interval = 10

rd = RefElemData(Line(),N; quad_rule_vol = gauss_lobatto_quad(0,0,N))
VX,EToV = uniform_mesh(Line(),K)
md_BCs = MeshData(VX,EToV,rd)
md = make_periodic(md_BCs,rd)

@unpack M,Dr,Vf,LIFT,Vq,Pq = rd

Qr = Pq'*(M*Dr)*Pq
B = diagm([-1.,1.])
E = Vf*Pq
QN = [Qr-Qr' E'*B;
      -B*E   0*B]
QNTr = sparse(QN')
Vh = [Vq; Vf]
VhP = Vh*Pq
Ph = M\Vh'
Nh = size(Vh,1)

ops = QN,QNTr,E,VhP,Ph

# fS(uL,uR) = @. (uL*uL + uL*uR + uR*uR)/6.0
# dS(uL,uR) = @. .5*max(abs(uL),abs(uR))*(uL-uR)
# fLF(uL,uR) = .5*(uL^2/2 + uR^2/2)

# function rhs(u,ops,rd,md)
#     @unpack rxJ,nxJ,mapP = md
#     QN,E,Vh,Ph = ops
#     Nf,Nq = size(E)
#     uf = @view u[Nq+1:end,:]
#     rhsQ = QN*u
#     rhsQ[Nq+1:Nq+Nf,:] .+= .5*(uf[mapP].*nxJ)
#     return rhsQ
# end
# function rhsf(u,ops,rd,md)
#     @unpack nxJ,mapP = md
#     QN,E,Vh,Ph = ops
#     Nf,Nq = size(E)
#     uf = @view u[Nq+1:end,:]
#     rhsQ = zeros(Nh,K)
#     rhsQ[Nq+1:end,:] .+= .5*(uf[mapP].*nxJ)
#     return rhsQ
# end
# function build_rhs_matrix(rhs::F,Np,K) where {F}
#     A = zeros(Np*K,Np*K)
#     u = zeros(Float64,Np,K)
#     for i = 1:Np*K
#         u[i] = 1.0
#         A[:,i] .= vec(rhs(u))
#         u[i] = 0.0
#     end
#     return droptol!(sparse(A),1e-12)
# end
# QNg = build_rhs_matrix(u->rhs(u,ops,rd,md),Nh,K)
# BNg = build_rhs_matrix(u->rhsf(u,ops,rd,md),Nh,K)
# QNgTr = -QNg # assume skew-symmetric

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

@inline avg(x,y) = .5*(x+y)
function fS(UL,UR)
    hL,huL = UL
    hR,huR = UR
    uL,uR = @. huL/hL, huR/hR
    f1 = @. avg(huL,huR)
    f2 = @. avg(huL,huR)*avg(uL,uR) + .5*hL*hR
    return SVector{2}(f1,f2)
end
function dS(UL,UR)
    hL,huL = UL
    hR,huR = UR
    uL,uR = @. huL/hL, huR/hR
    λL = @. abs(uL) + sqrt(abs(hL))
    λR = @. abs(uR) + sqrt(abs(hR))
    λ = @. .25*max(λL,λR)
    return SVector{2}(λ.*(hL.-hR),λ.*(huL.-huR))
end

function entropy_vars(U)
    h,hu = U
    u = @. hu/h
    v1 = @. h - .5*u^2
    v2 = u
    return SVector{2}(v1,v2)
end

function cons_vars(V)
    v1,v2 = V
    h = @. v1 + .5*v2^2
    hu = @. v2*h
    return SVector{2}(h,hu)
end

@unpack x = md
h = @. 2.0 + .0*(x > 0)
b = @. .5*sin(pi*x)
# b = @. 0*x
# bmax = maximum(b)
# h = @. 2.0 - b
hu = @. 0*h
Q = SVector{2}(h,hu)

dt0 = CFL*(x[2]-x[1]) / maximum(@. sqrt(abs(h)))
Nsteps = ceil(Int,T/dt0)
dt = T/Nsteps
ubounds = (0.0,maximum(h))

function rhs(Q,ops,rd,md,b)
    QN,QNTr,E,Vh,Ph = ops
    @unpack Dr,Vq,Pq = rd
    @unpack rxJ,nxJ,J,mapP = md
    Nf,Nq = size(E)

    Uh = cons_vars((x->VhP*x).(entropy_vars((x->Vq*x).(Q))))
    fids = Nq+1:Nq+Nf
    Uf = view.(Uh,(fids,fids),(:,:))
    UP = (x->view(x,mapP)).(Uf)
    rhsQ = ((x,y)->LIFT*(x.*nxJ+y)).(fS(Uf,UP), dS(Uf,UP))
    for e = 1:md.K
        Ue = view.(Uh,:,e)
        rhse = view.(rhsQ,:,e)
        ((x,y)->(x .+= rxJ[1,e]*Ph*y)).(rhse,hadsum(QNTr,fS,Ue))
        rhse[2] .+= Pq*(Ue[1][1:Nq].*(rxJ[1,e]*Vq*Dr*b[:,e]))
    end

    return (x->-x./J).(rhsQ)
end

@gif for i = 1:Nsteps

    rhs1 = rhs(Q,ops,rd,md,b)
    Qtmp1 = map((x,y)->x + dt*y,Q,rhs1)
    rhs2 = rhs(Qtmp1,ops,rd,md,b)
    rhs12 = @. .5*(rhs1 + rhs2)
    map((x,y)->x .= x + dt*y,Q,rhs12)

    if i%interval==0
        println("$i / $Nsteps: $ubounds, $ubounds, minh = $(minimum(Q[1]))")
        plot(rd.Vp*x,rd.Vp*Q[1],leg=false,ylims = (0,4))
        # scatter!(x,Q[2],ms=1,)
    end
end every interval

# # plot(x,Q[1],mark=:dot,ms=2,ylims = ubounds .+ (-.5,.5))
# # plot!(reshape(x,N+1,K),l,leg=false)
#
# # Δmass = sum(vec(MJ).*h)-init_mass
# # println("Δmass = $Δmass")
