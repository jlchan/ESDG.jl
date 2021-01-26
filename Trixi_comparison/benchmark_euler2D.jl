using BenchmarkTools
using EntropyStableEuler
using FluxDiffUtils
using LinearAlgebra
using NodesAndModes
using Plots
using Printf
using RecursiveArrayTools
using Setfield
using SparseArrays
using StartUpDG
using StaticArrays
using Test

function benchmark_euler(; initial_refinement_level=1, polydeg=3)
  include("euler2D_functions.jl")

  N = polydeg
  K1D = 2^initial_refinement_level

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

  SxTr,SyTr,Fmask,wq = precompute(rd,md)
  rhsQ = VectorOfArray(@SVector [zeros(size(md.xyz[1])) for i = 1:4])
  Q = initial_condition(md)
  return @benchmark rhs!($rhsQ.u,$Q,$md,$rd,$SxTr,$SyTr,$Fmask)
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
tabulate_benchmarks(benchmark_euler)
