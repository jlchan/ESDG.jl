using StartUpDG
using Plots
using RecursiveArrayTools
using StaticArrays

N = 3
K1D = 32
CFL = .5
FinalTime = 2.0

VX,EToV = uniform_mesh(Line(),K1D)
rd = RefElemData(Line(),N)
md = MeshData(VX,EToV,rd)
md = make_periodic(md,rd)

function rhs(Q,md::MeshData,rd::RefElemData)
    @unpack rxJ,J,nxJ,sJ,mapP = md
    @unpack Dr,LIFT,Vf = rd
    p,u = Q
    pf,uf = map(x->Vf*x,Q)
    pflux = @. .5*(uf[mapP]-uf)*nxJ - .5*(pf[mapP]-pf)*sJ
    uflux = @. .5*(pf[mapP]-pf)*nxJ - .5*(uf[mapP]-uf)*sJ
    rhspJ = rxJ.*(Dr*u) + LIFT*pflux
    rhsuJ = rxJ.*(Dr*p) + LIFT*uflux
    return VectorOfArray(SVector{2}(-rhspJ./J, -rhsuJ./J))
end

@unpack x = md
p = @. exp(-100*x^2)
u = zero.(p)
Q = VectorOfArray(SVector{2}([p,u]))

rk4a,rk4b,rk4c = ck45()
CN = (N+1)*(N+2)/2  # trace constant
dt = CFL * 2 / (CN*K1D)
Nsteps = convert(Int,ceil(FinalTime/dt))
dt = FinalTime/Nsteps

resQ = zero(Q)
for i = 1:Nsteps
    for INTRK = 1:5
        rhsQ = rhs(Q,md,rd)
        @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
        @. Q   += rk4b[INTRK]*resQ
    end

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
    end
end

p,u = Q
plot(rd.Vp*x,rd.Vp*p,ylims=(-.1,1.1),lw=2,leg=false)
scatter!(x,p)
