using NodesAndModes
using StartUpDG
using Plots
using LinearAlgebra
using ForwardDiff
using StaticArrays
using EntropyStableEuler
using Formatting
using RecursiveArrayTools

N = 5
K1D = 4
CFL = .05
FinalTime = 30.0

VX,VY,EToV = uniform_mesh(Quad(),K1D)
rd = RefElemData(Line(),N; quad_rule_vol = gauss_quad(0,0,2*N))
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

@unpack M,Dr,Ds,Vf,Vq,Pq = rd
Qr = M*Dr
Qs = M*Ds
invMQTr = Matrix(-M\Qr')
invMQTs = Matrix(-M\Qs')

function f(U)
    ρ,ρu,ρv,E = U
    u = ρu./ρ
    v = ρv./ρ
    p = pfun(Euler{2}(),U)
    f2x = @. ρu*u + p
    f3x = @. ρu*v
    f4x = @. (E + p)*u

    f2x = @. ρu*v
    f3x = @. ρv*v + p
    f4y = @. (E + p)*v
    return SVector{4}(ρu,f2x,f3x,f4x),SVector{4}(ρv,f2y,f3y,f4y)
end
fEC(uL,uR) = fS(Euler{2}(),uL,uR)
v_u(u) = v_ufun(Euler{2}(),u)
u_v(v) = u_vfun(Euler{2}(),v)
S(u) = Sfun(Euler{2}(),u)

function rhs(U,invMQTr,invMQTs,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack M,Vq,Pq,LIFT,Vf = rd

    ṽ = (x->Pq*x).(v_u((x->Vq*x).(U)))
    fN = (x->Pq*x).(f(u_v((x->Vq*x).(ṽ))))
    ff = (x->Vf*x).(fN)
    favg = (u->.5*(u[mapP]+u)).(ff)

    # correction: TODO fix
    uf = u_v((x->Vf*x).(ṽ))
    uP = (u->u[mapP]).(uf)
    fec = fEC(uP,uf)
    Δf = @. fec - favg
    Δv = v_u(uP) .- v_u(uf) # can also compute from ṽ

    # correction
    sign_Δv = map(x->sign.(x),Δv)
    Δf_dot_signΔv = sum(map((x,y)->x.*y.*nxJ,sign_Δv,Δf))
    ctmp = @. max(0,-Δf_dot_signΔv)
    c = map(x->ctmp.*x,sign_Δv) # central dissipation
    # c = map(x->nxJ .* x, -Δf) # EC

    λf = sqrt.(pfun(Euler{1}(),uf)*Euler{1}().γ ./ uf[1])
    λ = @. .5*max(λf,λf[mapP])
    LFu = map(x->λ .* x,uP .- uf)

    rhsJ(f,flux,c,LFu) = rxJ.*(invMQTr*f) + LIFT*(nxJ.*flux .- c .- LFu)
    return VectorOfArray((x->-x./J).(rhsJ.(fN,favg,c,LFu)))
end

@unpack x,y,J = md
wJq = diagm(rd.wq)*(Vq*J)

# shock initial conditions
ρ = @. 1.0 + exp(-25*(x^2+y^2))
u = @. .0 + 0*ρ
p = @. ρ^Euler{1}().γ
Q = VectorOfArray(prim_to_cons(Euler{1}(),(ρ,u,p)))
# E = 2*ones(size(ρ))
# Q = VectorOfArray(SVector{3}(ρ,ρu,E))

# sine wave
ρ = @. 1 + .5*sin(2*pi*(x+y))
# ρ = @. 1.0 + exp(-100*x^2)
u = @. .1 + 0*x
v = @. .2 + 0*x
p = @. 20. + 0*x

Q = VectorOfArray(prim_to_cons(Euler{2}(),(ρ,u,p)))

# rhsvec(u) = vec(-rhs(reshape(u,N+1,md.K),ops,md,rd))
# jac = ForwardDiff.jacobian(rhsvec,vec(u0.(x)))
# λ,V = eigen(jac)
# # scatter(λ,title="max real part = $(maximum(real(λ)))")
# val,id = findmax(real(λ))
# u = u0.(x) .+ 1e0*reshape(real(V[:,79]),N+1,md.K)

# setup
rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(FinalTime/dt))
dt = FinalTime/Nsteps
resu = zero.(Q)
unorm = zeros(Nsteps)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsu = rhs(Q.u,invMQTr,invMQTs,md,rd)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. Q   += rk4b[INTRK]*resu
    end

    unorm[i] = sum(wJq.*S((x->Vq*x).(Q.u)))

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

vv = rd.Vp*Q[1]
p1 = scatter(rd.Vp*x,rd.Vp*x,vv,zcolor=vv,leg=false)

ΔS = unorm[end]-unorm[1]
s = "ΔS = " * sprintf1("%1.1e",ΔS)
p2 = plot((1:Nsteps)*dt,unorm,leg=false,xlims = (0,FinalTime),title=s)
plot(p1,p2)
