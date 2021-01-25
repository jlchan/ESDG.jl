using Plots
using RecursiveArrayTools
using StartUpDG
using StaticArrays
using NodesAndModes
using Setfield

N = 3
K1D = 16
CFL = .25
FinalTime = .75

VX,VY,EToV = uniform_mesh(Quad(),K1D)

# make lumped lobatto element
r1D,w1D = gauss_lobatto_quad(0,0,N)
rq,sq = vec.(NodesAndModes.meshgrid(r1D))
wr,ws = vec.(NodesAndModes.meshgrid(w1D))
wq = @. wr*ws
rd = RefElemData(Quad(), N; quad_rule_vol=(rq,sq,wq), quad_rule_face=(r1D,w1D))
rd = @set rd.LIFT = droptol!(sparse(rd.LIFT),1e-12)
rd = @set rd.Vf = droptol!(sparse(rd.Vf),1e-12)

md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

function rhs(Q,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack Dr,Ds,LIFT,Vf = rd
    p,u,v = Q
    Qf = map(x->Vf*x,Q)
    dp,du,dv = map(x->x[mapP]-x,Qf)
    dUn = @. (du*nxJ + dv*nyJ)/sJ

    tau = 1.0
    pflux = @. .5*dUn - tau*.5*dp*sJ
    uflux = @. .5*dp*nxJ - tau*.5*dUn*nxJ
    vflux = @. .5*dp*nyJ - tau*.5*dUn*nyJ

    dudx = rxJ.*(Dr*u) + sxJ.*(Ds*u)
    dvdy = ryJ.*(Dr*v) + syJ.*(Ds*v)
    pr,ps = Dr*p, Ds*p
    dpdx = @. rxJ*pr + sxJ*ps
    dpdy = @. ryJ*pr + syJ*ps
    rhspJ = dudx + dvdy + LIFT*pflux
    rhsuJ = dpdx + LIFT*uflux
    rhsvJ = dpdy + LIFT*vflux
    return VectorOfArray(SVector{3}(-rhspJ./J, -rhsuJ./J, -rhsvJ./J))
end

@unpack x,y = md
p = @. exp(-100*(x^2+y^2))
u = zero(p)
v = zero(p)
Q = VectorOfArray(SVector{3}([p,u,v]))

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

p,u,v = Q
scatter(map(x->rd.Vp*x,md.xyz)...,rd.Vp*p,zcolor=rd.Vp*p,lw=2,leg=false,cam=(0,90),msw=0)
