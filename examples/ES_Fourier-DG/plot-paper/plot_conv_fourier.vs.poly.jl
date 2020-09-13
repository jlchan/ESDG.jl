using Revise # reduce recompilation time
using Plots
# using Documenter
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using UnPack

push!(LOAD_PATH, "./src")
using CommonUtils
using Basis1D
using Basis2DQuad
using UniformQuadMesh

using SetupDG

push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

N =
x = 0:N:2*pi


h       = 2*pi/N
column  = [0; .5*(-1).^(1:Np_F-1).*cot.((1:Np_F-1)*h/2)]
Dt      = Array{Float64,2}(Toeplitz(column,column[[1;Np_F:-1:2]]))
