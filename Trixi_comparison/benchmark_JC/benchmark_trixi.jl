#!/usr/bin/evj julia
# Benchmark some common RHS evaluations on a uniform 2D mesh in Trixi.jl

using Pkg
Pkg.activate(temp=true)
Pkg.add("BenchmarkTools")
Pkg.add("Trixi")

using LinearAlgebra
BLAS.set_num_threads(Threads.nthreads())

using Printf

using BenchmarkTools
using Trixi


# Use trivial constant initial conditions for simple performance measurements
function initial_condition(x, t, equation::LinearScalarAdvectionEquation2D)
  return Trixi.SVector(2.0)
end

function initial_condition(x, t, equation::CompressibleEulerEquations2D)
  ϱ   = 1.0
  ϱv1 = 0.1
  ϱv2 = -0.2
  ϱe  = 10.0
  return Trixi.SVector(ϱ, ϱv1, ϱv2, ϱe)
end



function benchmark_linadv(; initial_refinement_level=1, polydeg=3)

  advectionvelocity = (0.1, -0.2)
  equations = LinearScalarAdvectionEquation2D(advectionvelocity)

  surface_flux = flux_lax_friedrichs
  solver = DGSEM(polydeg, surface_flux)

  coordinates_min = (-1.0, -1.0)
  coordinates_max = ( 1.0,  1.0)
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=initial_refinement_level,
                  n_cells_max=100_000)

  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

  t0 = 0.0
  u0 = compute_coefficients(t0, semi)
  du = similar(u0)

  @benchmark Trixi.rhs!($du, $u0, $semi, $t0)
end


function benchmark_euler(; initial_refinement_level=1, polydeg=3)

  γ = 1.4
  equations = CompressibleEulerEquations2D(γ)

  surface_flux = flux_ranocha
  volume_flux  = flux_ranocha
  solver = DGSEM(polydeg, surface_flux, VolumeIntegralFluxDifferencing(volume_flux))

  coordinates_min = (-1.0, -1.0)
  coordinates_max = ( 1.0,  1.0)
  mesh = TreeMesh(coordinates_min, coordinates_max,
                  initial_refinement_level=initial_refinement_level,
                  n_cells_max=100_000)

  semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

  t0 = 0.0
  u0 = compute_coefficients(t0, semi)
  du = similar(u0)

  @benchmark Trixi.rhs!($du, $u0, $semi, $t0)
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
# tabulate_benchmarks(benchmark_linadv)
tabulate_benchmarks(benchmark_euler; polydeg=7)
