using NodesAndModes
using StartUpDG
using Plots
using LinearAlgebra
using ForwardDiff

N = 7
K1D = 10
CFL = .1
FinalTime = 1.0

VX,EToV = uniform_mesh(Line(),K1D)
rd = RefElemData(Line(),N; quad_rule_vol = gauss_quad(0,0,2*N-1))
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

@unpack M,Dr,Vf,Vq,Pq = rd
Qr = M*Dr
invMQTr = Matrix(-M\Qr')
ops = invMQTr,Vf,M,Vq,Pq

f(u) = u^2/2
fEC(uL,uR) = (uL^2 + uL*uR + uR^2) / 6

function rhs(u,ops,md::MeshData,rd::RefElemData)
    @unpack rxJ,J,nxJ,mapP = md
    @unpack Dr,LIFT,Vf = rd
    invMQTr,Vf,M,Vq,Pq = ops

    fN = Pq*f.(Vq*u)
    ff = Vf*fN
    favg = .5*(ff[mapP]+ff)

    # correction
    uf = Vf*u
    Δf = @. fEC(uf[mapP],uf)-favg
    Δu = uf[mapP]-uf
    c = @. max(0,-Δf*nxJ*sign(Δu))*sign(Δu)
    # c = @. -Δf*nxJ # EC
    # c = 0 # central

    rhsuJ = rxJ.*(invMQTr*fN) + LIFT*(nxJ.*favg .- c)
    return -rhsuJ./J
end

@unpack x,J = md
wJq = diagm(rd.wq)*(Vq*J)

# shock initial conditions
u = @. exp(-25*x^2)
# u = @. .1-.1*sin(pi*x)
b = zero(u)

# steady state
u0(x) = @. 2 + sin(8*pi*(x-.7))
u = @. u0(x) + 1e-3*cos(pi*x)
b = -rhs(u0.(x),ops,md,rd)

rhsvec(u) = vec(-rhs(reshape(u,rd.N+1,md.K),ops,md,rd))
jac = ForwardDiff.jacobian(rhsvec,vec(u0.(x)))
λ,V = eigen(jac)
# scatter(λ,title="max real part = $(maximum(real(λ)))")
val,id = findmax(real(λ))
u = u0.(x) .+ 1e0*reshape(real(V[:,79]),rd.N+1,md.K)

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

    unorm[i] = sum(wJq.*(Vq*u).^2)

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

plot(rd.Vp*x,rd.Vp*u - u0.(rd.Vp*x),lw=2,leg=false,lcolor=:black)

# plot((1:Nsteps)*dt,unorm,leg=false)
