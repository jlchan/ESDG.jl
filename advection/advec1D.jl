using StartUpDG
using Plots

N = 3
K1D = 32
CFL = .5
FinalTime = 2.0

VX,EToV = uniform_mesh(Line(),K1D)
rd = RefElemData(Line(),N)
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

function rhs(u,md::MeshData,rd::RefElemData)
    @unpack rxJ,J,nxJ,mapP = md
    @unpack Dr,LIFT,Vf = rd
    uf = Vf*u
    uflux = .5*(uf[mapP]-uf)
    rhsuJ = rxJ.*(Dr*u) + LIFT*(nxJ.*uflux)
    return -rhsuJ./J
end

@unpack x = md
u = @. exp(-25*x^2)

rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(FinalTime/dt))
dt = FinalTime/Nsteps

resu = zero(u)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsu = rhs(u,md,rd)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

plot(rd.Vp*x,rd.Vp*u,ylims=(-.1,1.1),lw=2,leg=false)
scatter!(x,u)
