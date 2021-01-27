using BenchmarkTools
using EntropyStableEuler
using FluxDiffUtils
using LinearAlgebra
using NodesAndModes
using Printf
using RecursiveArrayTools
using Setfield
using SparseArrays
using StartUpDG
using StaticArrays
using Test

BLAS.set_num_threads(Threads.nthreads())

function precompute(rd::RefElemData,md::MeshData)
    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,J,sJ,mapP  = md
    @unpack Dr,Ds,M,LIFT,Vf = rd

    # precompute for Cartesian mesh
    Qr,Qs = M*Dr, M*Ds
    Qx = @. rxJ[1]*Qr + sxJ[1]*Qs
    Qy = @. ryJ[1]*Qr + syJ[1]*Qs
    SxTr,SyTr = map(A->droptol!(sparse(transpose(M\(A-A'))),1e-12),(Qx,Qy)) # flip sign for skew transpose

    # build Fmask to extract face ids
    Vf_ids = findall(@. abs(Vf) > 1e-12)
    Fmask = zeros(Int64,size(Vf,1))
    for ij in Vf_ids
        Fmask[ij[1]] = ij[2]
    end

    vmapM = reshape(1:length(md.x),size(md.x)...)[Fmask,:]
    vmapP = vmapM[mapP]

    r1D,w1D = gauss_lobatto_quad(0,0,N)
    Lscale = 1.0 / w1D[1]

    return SxTr,SyTr,Fmask,vmapM,vmapP,Lscale
end

function compute_logs(Q)
    Qprim = cons_to_prim_beta(Euler{2}(),Q)  # WARNING: global. Why is it faster than the no-alloc version though?
    return (Qprim[1],Qprim[2],Qprim[3],Qprim[4],log.(Qprim[1]),log.(Qprim[4])) # append logs
end

# function compute_logs!(Qlog,Q)
#     for i = 1:length(first(Q.u))
#         Qprim = cons_to_prim_beta(Euler{2}(),getindex.(Q.u,i))
#         Qlogi = (Qprim[1],Qprim[2],Qprim[3],Qprim[4],log(Qprim[1]),log(Qprim[4]))
#         setindex!.(Qlog,Qlogi,i)
#     end
#     return nothing
# end

f2D(QL,QR) = fS_prim_log(Euler{2}(),QL,QR)

# compute surface flux + lifted contributions
function surface_contributions!(rhsQ,Qlog,rd,md,Fmask)
    @unpack LIFT = rd
    @unpack mapP,nxJ,nyJ = md
    Qflog = map(x->x[Fmask,:],Qlog) # WARNING: global
    QPlog = map(x->x[mapP],Qflog)   # WARNING: global
    flux = @SVector [zeros(size(nxJ,1)) for fld = 1:4] # tmp storage
    for e = 1:size(nxJ,2)
        for i = 1:size(nxJ,1)
            fi = f2D(getindex.(Qflog,i,e),getindex.(QPlog,i,e))
            fni = nxJ[i].*fi[1] + nyJ[i].*fi[2]
            setindex!.(flux,fni,i)
        end
        map((x,y)->x .+= LIFT*y,view.(rhsQ,:,e),flux) # WARNING: LIFT can be optimized
    end
    return nothing
end

function applyLIFT!(x,Fmask,Lscale,xf)
    for (idf,idv) in enumerate(Fmask)
       x[idv] += xf[idf]*Lscale
    end
    return nothing
end

function surface_contributions2!(rhsQ,Qlog,rd,md,Fmask,vmapM,vmapP,Lscale)
    @unpack LIFT = rd
    @unpack mapP,nxJ,nyJ = md

    flux = @SVector [zeros(size(nxJ,1)) for fld = 1:4] # tmp storage
    for e = 1:size(nxJ,2)
        for i = 1:size(nxJ,1)
            fi = f2D(getindex.(Qlog,vmapM[i,e]),getindex.(Qlog,vmapP[i,e]))
            fni = nxJ[i].*fi[1] + nyJ[i].*fi[2]
            setindex!.(flux,fni,i)
        end
        # map((x,y)->x .+= LIFT*y,view.(rhsQ,:,e),flux) # WARNING: LIFT can be optimized
        map((x,xf)->applyLIFT!(x,Fmask,Lscale,xf),view.(rhsQ,:,e),flux)
    end
    return nothing
end

function volume_contributions!(rhsQ,Qlog,SxTr,SyTr)
    for e = 1:size(first(Qlog),2)
        hadamard_sum_ATr!(view.(rhsQ,:,e),(SxTr,SyTr),f2D,view.(Qlog,:,e)) # overwrites
    end
    return nothing
end

function rhs!(rhsQ,Q,md,rd,precomp)
    SxTr,SyTr,Fmask,vmapM,vmapP,Lscale = precomp
    fill!.(rhsQ,zero(eltype(Q)))
    Qlog = compute_logs(Q)
    # surface_contributions!(rhsQ,Qlog,rd,md,Fmask)
    surface_contributions2!(rhsQ,Qlog,rd,md,Fmask,vmapM,vmapP,Lscale)
    volume_contributions!(rhsQ,Qlog,SxTr,SyTr)
    (x-> x ./= -md.J[1,1]).(rhsQ)
end

function initial_condition(md)
    @unpack x,y = md
    rho = @. 2 + .0*exp(-25*(x^2+y^2))
    # rho = @. 2 + .1*sin(pi*x)*sin(pi*y)
    u = .5*ones(size(x))
    v = 0*ones(size(x))
    p = one.(rho) # rho.^Euler{2}().γ
    Q = VectorOfArray(SVector{4}(prim_to_cons(Euler{2}(),(rho,u,v,p))...))
    return Q
end

function benchmark_euler(; initial_refinement_level=1, polydeg=3)
  N = polydeg
  K1D = 2^initial_refinement_level

  # make sparse operators
  r1D,w1D = gauss_lobatto_quad(0,0,N)
  rq,sq = vec.(NodesAndModes.meshgrid(r1D))
  wr,ws = vec.(NodesAndModes.meshgrid(w1D))
  wq = @. wr*ws
  rd = RefElemData(Quad(), N;
          quad_rule_vol=(rq,sq,wq), quad_rule_face=(r1D,w1D))
  rd = @set rd.LIFT = droptol!(sparse(rd.LIFT),1e-12)
  rd = @set rd.Vf = droptol!(sparse(rd.Vf),1e-12)

  VX,VY,EToV = uniform_mesh(Quad(),K1D)
  md_nonperiodic = MeshData(VX,VY,EToV,rd)
  md = make_periodic(md_nonperiodic,rd)

  precomp = precompute(rd,md)
  rhsQ = VectorOfArray(@SVector [zeros(size(md.xyz[1])) for i = 1:4])
  Q = initial_condition(md)
  return @benchmark rhs!($rhsQ.u,$Q,$md,$rd,$precomp)
end

function run_benchmarks(benchmark_run; levels=0:5, polydeg=3)
  runtimes = zeros(length(levels))
  for (idx,initial_refinement_level) in enumerate(levels)
    result = benchmark_run(; initial_refinement_level, polydeg)
    display(result)
    runtimes[idx] = result |> median |> time # in nanoseconds
  end
  return (; levels, runtimes, polydeg)
end

function tabulate_benchmarks(args...; kwargs...)
  result = run_benchmarks(args...; kwargs...)
  println("#Elements | Runtime in seconds")
  for (level,runtime) in zip(result.levels, result.runtimes)
    @printf("%9d | %.2e\n", 4^level, 1.0e-9 * runtime)
  end
end

versioninfo(verbose=true)
tabulate_benchmarks(benchmark_euler; levels=0:5)
