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

VX,EToV = uniform_mesh(Line(),K1D)
rd = RefElemData(Line(),N; quad_rule_vol = gauss_quad(0,0,2*N))
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

rd_overint = RefElemData(Line(),N; quad_rule_vol = gauss_quad(0,0,2*N+2))

@unpack M,Dr,Vf,Vq,Pq = rd
Qr = M*Dr
invMQTr = Matrix(-M\Qr')

function f(U)
    ρ,ρu,E = U
    u = ρu./ρ
    p = pfun(Euler{1}(),U)
    fm = @. ρu*u + p
    fE = @. (E + p)*u
    return SVector{3}(ρu,fm,fE)
end
fEC(uL,uR) = fS(Euler{1}(),uL,uR)
v_u(u) = v_ufun(Euler{1}(),u)
u_v(v) = u_vfun(Euler{1}(),v)
S(u) = Sfun(Euler{1}(),u)

function rhs(U,invMQTr,md::MeshData,rd::RefElemData,rd_overint::RefElemData)
    @unpack rxJ,J,nxJ,mapP = md
    @unpack M,Vq,Pq,Dr,LIFT,Vf = rd

    ṽ = (x->Pq*x).(v_u((x->Vq*x).(U)))
    fN = (x->Pq*x).(f(u_v((x->Vq*x).(ṽ))))
    ff = (x->Vf*x).(fN)
    favg = (u->.5*(u[mapP]+u)).(ff)

    # correction
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

@unpack x,J = md
wJq = diagm(rd.wq)*(Vq*J)

# sine wave
ρ = @. 1 + .5*sin(2*pi*x)
u = @. .2 + 0*x
p = @. 20. + 0*x

# # non-dimensionalization
# p_ref = maximum(p)
# ρ_ref = minimum(ρ)
# a_ref = sqrt(Euler{1}().γ*p_ref/ρ_ref)
# ρ = ρ/ρ_ref
# u = u/a_ref
# p = p/p_ref
# T = T*a_ref

Q = VectorOfArray(prim_to_cons(Euler{1}(),(ρ,u,p)))

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
        rhsu = rhs(Q.u,invMQTr,md,rd,rd_overint)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. Q   += rk4b[INTRK]*resu
    end

    unorm[i] = sum(wJq.*S((x->Vq*x).(Q.u)))

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

p1 = plot(rd.Vp*x,rd.Vp*Q[1],lw=2,leg=false,lcolor=:black)

ΔS = unorm[end]-unorm[1]
s = "ΔS = " * sprintf1("%1.1e",ΔS)
p2 = plot((1:Nsteps)*dt,unorm,leg=false,xlims = (0,FinalTime),title=s)
plot(p1,p2)
# plot(p1)
