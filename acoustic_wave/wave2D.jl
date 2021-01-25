using Plots
using RecursiveArrayTools
using StartUpDG
using StaticArrays
using NodesAndModes

N = 3
K1D = 32
CFL = .5
FinalTime = .75

VX,VY,EToV = uniform_mesh(Tri(),K1D)
rd = RefElemData(Tri(),N)
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

function rhs(Q,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack Dr,Ds,LIFT,Vf = rd
    p,u,v = Q
    Qf = map(x->Vf*x,Q)
    dp,du,dv = map(x->x[mapP]-x,Qf)
    dUn = @. du*nxJ + dv*nyJ

    tau = 1
    pflux = @. .5*dUn - tau*.5*dp*sJ
    uflux = @. .5*dp*nxJ - tau*.5*dUn*nxJ/sJ
    vflux = @. .5*dp*nyJ - tau*.5*dUn*nyJ/sJ

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

# V1 = vandermonde(Tri(),N,nodes(Tri(),1)...)/rd.VDM
# t = EToV'
# function to_rgba(x::UInt32)
#     a = ((x & 0xff000000)>>24)/255
#     b = ((x & 0x00ff0000)>>16)/255
#     g = ((x & 0x0000ff00)>>8)/255
#     r = (x & 0x000000ff)/255
#     RGBA(r, g, b, a)
# end
# function triplot(t,xyz...)
#     img = to_rgba.(Triplot.rasterize(xyz...,t)')
#     plot(map(x->[extrema(x)...],xyz[1:end-1])...,img,yflip=false)
# end
