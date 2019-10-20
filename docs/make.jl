push!(LOAD_PATH, pwd()*"/src/")   # TODO: Refactor module into folder? Also path is
push!(LOAD_PATH, pwd()*"/src/Utils")
using Documenter
using Utils, MeshUtils

makedocs(modules=[Utils, MeshUtils], sitename="Documentation for 2D DG code")
