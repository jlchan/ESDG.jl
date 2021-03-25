using NodesAndModes
using StartUpDG
using Plots
using LinearAlgebra
using ForwardDiff
using EntropyStableEuler
using Formatting

N = 4
K1D = 20
CFL = .25
FinalTime = 2.0

VX,EToV = uniform_mesh(Line(),K1D)
rd = RefElemData(Line(),N; quad_rule_vol = gauss_quad(0,0,N+2))
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

@unpack M,Dr,Vf,Vq,Pq = rd
Qr = M*Dr
invMQTr = Matrix(-M\Qr')
ops = invMQTr,Vf,M,Vq,Pq

f(u) = u
u_v(v) = exp(v)
v_u(u) = log(u)
S(u) = u*log(u) - u
fEC(uL,uR) = logmean(uL,uR)

# f(u) = u
# u_v(v) = v
# v_u(u) = u
# S(u) = u^2/2
# fEC(uL,uR) = .5*(uL+uR)

# f(u) = u^2/2
# S(u) = u^2/2
# u_v(v) = v
# v_u(u) = u
# fEC(uL,uR) = (uL^2 + uL*uR + uR^2)/6

function rhs(u,ops,md::MeshData,rd::RefElemData)
    @unpack rxJ,J,nxJ,mapP = md
    @unpack Dr,LIFT,Vf = rd
    invMQTr,Vf,M,Vq,Pq = ops

    ṽ = Pq*v_u.(Vq*u)
    fN = Pq*f.(u_v.(Vq*ṽ))
    ff = Vf*fN
    favg = .5*(ff[mapP]+ff)

    # correction
    uf = u_v.(Vf*ṽ)
    Δf = @. fEC(uf[mapP],uf)-favg
    Δv = v_u.(uf[mapP])-v_u.(uf) # can also compute from ṽ
    c = @. max(0,-Δf*nxJ*sign(Δv))*sign(Δv)
    # c = @. -Δf*nxJ # EC
    # c = 0 # central

    rhsuJ = rxJ.*(invMQTr*fN) + LIFT*(nxJ.*favg .- c)
    return -rhsuJ./J
end

@unpack x,J = md
wJq = diagm(rd.wq)*(Vq*J)

# shock initial conditions
u = @. 1.0 + exp(-25*x^2)
u = @. .25 + exp(sin(pi*x))
u = @. 1 + .5*(x > 0)
# u = @. .5-.5*sin(pi*x)
b = zero(u)

# # steady state
# u0(x) = @. 2 + sin(8*pi*(x-.7))
# u = @. u0(x) + 1e-3*cos(pi*x)
# b = -rhs(u0.(x),ops,md,rd)

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
resu = zero(u)
unorm = zeros(Nsteps)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsu = b + rhs(u,ops,md,rd)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    unorm[i] = sum(wJq.*S.(Vq*u))

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

p1 = plot(rd.Vp*x,rd.Vp*u,lw=2,leg=false,lcolor=:black)
# plot(rd.Vp*x,rd.Vp*u - u0.(rd.Vp*x),lw=2,leg=false,lcolor=:black)

ΔS = unorm[end]-unorm[1]
s = "ΔS = " * sprintf1("%1.1e",ΔS)
p2 = plot((1:Nsteps)*dt,unorm,leg=false,xlims = (0,T),title=s)
plot(p1,p2)
