using NodesAndModes
using StartUpDG
using UnPack
using LinearAlgebra
using SparseArrays
using FluxDiffUtils
using StaticArrays
using Setfield
using Plots

N = 7
K = 16
T = 2
CFL = .025

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
E = M*LIFT
invm = 1 ./ diag(M)

Qskew = Matrix(Q-Q')

fS(uL,uR) = @. (uL*uL + uL*uR + uR*uR)/6.0
dS(uL,uR) = @. .5*max(abs(uL),abs(uR))*(uL-uR)
fLF(uL,uR) = @. .5*(uL^2/2 + uR^2/2)

function hadsum(ATr,fS::F,u) where {F}
    rhs = similar(u)
    n = length(u)
    for i = 1:n
        ui = u[i]
        val = 0.0
        for j = 1:n
            val += ATr[j,i]*fS(ui,u[j])
        end
        rhs[i] = val
    end
    return rhs
end

function rhs(u,invm,Qskew,md,rd)
    @unpack Vf,LIFT = rd
    @unpack J,rxJ,nxJ,mapP = md
    rhsu = similar(u)
    for e = 1:md.K
        rhsu[:,e] .= invm.*hadsum(-Qskew,fS,u[:,e])
    end

    uf = Vf*u
    uP = uf[mapP]
    rhsu += LIFT*(fLF(uP,uf).*nxJ - .5*dS(uP,uf))
    return -rhsu./J
end

@unpack x,J = md
u = @. (x > -1/3)*1.0
resu = similar(u)
ulims = extrema(u) .+ (-.5,.5)

dt = CFL*(x[2]-x[1]) / maximum(abs.(u))
Nsteps = ceil(Int,T/dt)
dt = T/Nsteps

energy = Float64[]
@gif for i = 1:Nsteps
    global u
    r = rhs(u,invm,Qskew,md,rd)
    @. u = u + dt*r
    push!(energy,sum(M*(u.^2 .*J)))
    if i%interval == 0
        plot(x,u,mark=:dot,leg=false,ylims=ulims)
        println("on $i / $Nsteps")
    end
end every interval
