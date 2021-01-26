using BenchmarkTools
using EntropyStableEuler
using FluxDiffUtils
using LinearAlgebra
using NodesAndModes
using Plots
using RecursiveArrayTools
using Setfield
using SparseArrays
using StartUpDG
using StaticArrays
using Test

N = 3
K1D = 16
CFL = .5
FinalTime = 1.0

function lobatto_quad_2D(N)
    # make lumped lobatto element
    r1D,w1D = gauss_lobatto_quad(0,0,N)
    rq,sq = vec.(NodesAndModes.meshgrid(r1D))
    wr,ws = vec.(NodesAndModes.meshgrid(w1D))
    wq = @. wr*ws
    return rq,sq,wq
end

rq,sq,wq = lobatto_quad_2D(N)
rd = RefElemData(Quad(), N; quad_rule_vol=(rq,sq,wq), quad_rule_face=(r1D,w1D))
rd = @set rd.LIFT = droptol!(sparse(rd.LIFT),1e-12)
rd = @set rd.Vf = droptol!(sparse(rd.Vf),1e-12)

VX,VY,EToV = uniform_mesh(Quad(),K1D)
md_nonperiodic = MeshData(VX,VY,EToV,rd)
md = make_periodic(md_nonperiodic,rd)

@unpack x,y = md
rho = @. 2 + .0*exp(-25*(x^2+y^2))
# rho = @. 2 + .1*sin(pi*x)*sin(pi*y)
u = ones(size(x))
v = .0*randn(size(x))
p = one.(rho) # rho.^Euler{2}().γ
Q = VectorOfArray(SVector{4}(prim_to_cons(Euler{2}(),(rho,u,v,p))...))

function precompute(rd::RefElemData,md::MeshData)
    # @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ
    @unpack J,sJ,mapP = md
    # @unpack Dr,Ds = rd
    @unpack M,LIFT,Vf = rd

    # not actually unit vectors, just predivide by J
    nx = md.nxyzJ[1] ./ J[1,1]
    ny = md.nxyzJ[2] ./ J[1,1]

    # precompute for Cartesian mesh
    Qr,Qs = M*rd.Drst[1], M*rd.Drst[2]
    Qx = @. (md.rstxyzJ[1,1][1]*Qr + md.rstxyzJ[1,2][1]*Qs) / J[1,1]
    Qy = @. (md.rstxyzJ[2,1][1]*Qr + md.rstxyzJ[2,2][1]*Qs) / J[1,1]
    SxTr,SyTr = map(A->droptol!(sparse(transpose(M\(A-A'))),1e-12),(Qx,Qy)) # flip sign for skew transpose

    # build Fmask
    Vf_ids = findall(@. abs(Vf) > 1e-12)
    Fmask = zeros(Int64,size(Vf,1))
    for ij in Vf_ids
        Fmask[ij[1]] = ij[2]
    end

    return nx,ny,Qx,Qy,SxTr,SyTr,Fmask,LIFT,diag(M),mapP
end

function compute_logs(Q)
    Qprim = cons_to_prim_beta(Euler{2}(),Q)  # WARNING: global
    return (Qprim[1],Qprim[2],Qprim[3],Qprim[4],log.(Qprim[1]),log.(Qprim[4])) # append logs
end

f2D(QL,QR) = fS_prim_log(Euler{2}(),QL,QR)

# compute surface flux + lifted contributions
function surface_contributions!(rhsQ,Qlog,rd,md,Fmask,nx,ny)
    @unpack LIFT = rd
    @unpack mapP = md
    Qflog = map(x->x[Fmask,:],Qlog) # WARNING: global
    QPlog = map(x->x[mapP],Qflog)   # WARNING: global
    flux = @SVector [zeros(size(nx,1)) for fld = 1:4] # tmp storage
    for e = 1:size(nx,2)
        for i = 1:size(nxJ,1)
            fi = f2D(getindex.(Qflog,i,e),getindex.(QPlog,i,e))
            fni = nx[i].*fi[1] + ny[i].*fi[2]
            setindex!.(flux,fni,i)
        end
        map((x,y)->x .+= LIFT*y,view.(rhsQ,:,e),flux) # WARNING: LIFT can be optimized
    end
    return nothing
end

function volume_contributions!(rhsQ,Qlog,SxTr,SyTr)
    for e = 1:size(first(Qlog),2)
        hadamard_sum_ATr!(view.(rhsQ,:,e),(SxTr,SyTr),f2D,view.(Qlog,:,e)) # overwrites
    end
    return nothing
end

function rhs(Q,md::MeshData,rd::RefElemData)
    nx,ny,Qx,Qy,SxTr,SyTr,Fmask,LIFT,wq,mapP = precompute(rd,md)
    rhsQ = @SVector [zeros(Float64,size(md.xyz[1])) for i = 1:4]
    Qlog = compute_logs(Q)
    surface_contributions!(rhsQ,Qlog,rd,md,Fmask,nx,ny)
    volume_contributions!(rhsQ,Qlog,SxTr,SyTr)
    return VectorOfArray((x-> -1.0 .*x).(rhsQ))
end

function rhs!(rhsQ,Q,md,rd)
    Qlog = compute_logs(Q)
    surface_contributions!(rhsQ,Qlog,rd,md,Fmask,nx,ny)
    volume_contributions!(rhsQ,Qlog,SxTr,SyTr)
    return VectorOfArray((x-> -1.0 .*x).(rhsQ))
end

nx,ny,Qx,Qy,SxTr,SyTr,Fmask,LIFT,wq,mapP = precompute(rd,md)
rhsQ = @SVector [zeros(Float64,size(md.xyz[1])) for i = 1:4]
Qlog = compute_logs(Q)
# @btime rhs!($rhsQ,$Q,$md,$rd)

# rk4a,rk4b,rk4c = ck45()
# CN = (N+1)*(N+2)  # trace constant
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
