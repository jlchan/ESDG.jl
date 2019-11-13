push!(LOAD_PATH, "./src/")   # TODO: Refactor module into folder? Also path is
push!(LOAD_PATH, "./src/Utils/")
using Documenter
using Utils
using Basis1D,Basis2DTri,Basis2DQuad
using UniformQuadMesh, UniformTriMesh

makedocs(modules=[Utils,Basis1D,Basis2DTri,Basis2DQuad,UniformTriMesh], sitename="Documentation for JuliaDG")
