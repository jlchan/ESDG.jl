using Test, SafeTestsets

@safetestset "Triangulate.jl interface tests" begin
    include("triangulate_tests.jl")
end

@safetestset "SBP tests" begin
    include("sbp_tests.jl")
end

@safetestset "Flux differencing" begin
    include("fluxdiff_tests.jl")
end