push!(LOAD_PATH, "./src/")   # TODO: Refactor module into folder? Also path is
push!(LOAD_PATH, "./src/CommonUtils/")
using Documenter
using CommonUtils
using Basis1D,Basis2DTri,Basis2DQuad,Basis3DHex
using UniformQuadMesh,UniformTriMesh,UniformHexMesh

makedocs(modules=[CommonUtils,Basis1D,Basis2DTri,UniformTriMesh], sitename="Documentation for JuliaDG")
