using EntropyStableEuler
using FluxDiffUtils
using NodesAndModes
using Plots
using RecursiveArrayTools
using Setfield
using SparseArrays
using StartUpDG
using StaticArrays

using TimerOutputs
# const to = TimerOutput()
reset_timer!(to)

N = 4
K1D = 1
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

@unpack r,s = rd
rho = @. 2 + 0*exp(-100*(r^2+s^2))
u = .0*randn(size(r))
v = .0*randn(size(s))
p = rho.^Euler{2}().γ
Q = VectorOfArray(SVector{4}(prim_to_cons(Euler{2}(),(rho,u,v,p))...))

function rhs(Q,md::MeshData,rd::RefElemData)
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack M,Dr,Ds,LIFT,Vf = rd

    # not actually unit vectors, just predivide by J
    nx, ny = nxJ./(Vf*J), nyJ./(Vf*J)

    # precompute for Cartesian mesh
    Qr,Qs = M*Dr, M*Ds
    Qx = @. (rxJ[1,1]*Qr + sxJ[1,1]*Qs)/J[1,1]
    Qy = @. (ryJ[1,1]*Qr + syJ[1,1]*Qs)/J[1,1]
    # SxTr,SyTr = map(A->droptol!(sparse(-M\(A-A')),1e-11),(Qx,Qy)) # flip sign for skew transpose
    SxTr,SyTr = map(A->Matrix(-M\(A-A')),(Qx,Qy))

    @timeit to "compute logs" begin
        compute_logs(Q) = map(x->log.(x),(first(Q),last(Q)))
        Qprim = cons_to_prim_beta(Euler{2}(),Q)
        Qlog = (Qprim...,compute_logs(Qprim)...) # append logs
    end

    @timeit to "surface fluxes" begin
        @timeit to "interp" begin
            # compute surface flux + lifted contributions
            Qflog = map(x->Vf*x,Qlog) # assumes Vf*u = u[face_indices]!
            QPlog = map(x->x[mapP],Qflog)
        end
        f2D(QL,QR) = fS_prim_log(Euler{2}(),QL,QR)
        @timeit to "flux" flux = f2D(Qflog,QPlog)
        @timeit to "flux dot" fSn  = map((fx,fy)->(@. fx*nx + fy*ny), flux...)
        @timeit to "lift" rhsQ = map(x->LIFT*x,fSn)
    end

    @timeit to "volume flux diff" begin
        # compute volume contributions
        accum_col!(x,y,e) = x[:,e] .+= y
        rhstmp = zero.(getindex.(Qprim,:,1)) # temp storage
        for e = 1:md.K
            @timeit to "Qe" Qe = getindex.(Qlog,:,e)
            @timeit to "hadamard" hadamard_sum_ATr!(rhstmp,(SxTr,SyTr),f2D,Qe)
            @timeit to "accum" accum_col!.(rhsQ,rhstmp,e)
        end
    end

    return rhsQ
end

for i = 1:10
    rhs(Q,md,rd);
end
show(to)


# norm.(rhs(Q,md,rd))
#@btime rhs($Q,$md,$rd)

# rk4a,rk4b,rk4c = ck45()
# CN = (N+1)*(N+2)/2  # trace constant
# dt = CFL * 2 / (CN*K1D)
# Nsteps = convert(Int,ceil(FinalTime/dt))
# dt = FinalTime/Nsteps
#
# resQ = zero(Q)
# for i = 1:Nsteps
#     for INTRK = 1:5
#         rhsQ = rhs(Q,md,rd)
#         @. resQ = rk4a[INTRK]*resQ + dt*rhsQ
#         @. Q   += rk4b[INTRK]*resQ
#     end
#
#     if i%100==0 || i==Nsteps
#         println("Number of time steps $i out of $Nsteps")
#     end
# end
#
# zp = rd.Vp*Q[1]
# scatter(map(x->rd.Vp*x,md.xyz)...,zp,zcolor=zp,lw=2,leg=false,cam=(0,90),msw=0)
