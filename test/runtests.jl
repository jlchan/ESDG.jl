using ESDG, Test
using LinearAlgebra
using StaticArrays
using StructArrays

@testset "Triangulate.jl interface tests" begin
    include("triangulate_tests.jl")
end

@testset "SBP tests" begin
    include("sbp_tests.jl")
end

@testset "Flux differencing" begin
    # test scalars
    u = randn(4)
    du = similar(u)
    A = Matrix{Float64}(reshape(1:4*4,4,4))
    hadamard_sum_ATr!(du,A',(x,y)->(x+y),u)
    target = A*u + vec(u.*sum(A,dims=2))
    @test du ≈ target

    # test structarrays
    v = copy(u)
    dv = similar(v)
    Q = StructArray{SVector{2,Float64}}((u,v))
    dQ = similar(Q)    
    hadamard_sum_ATr!(dQ,A',(x,y)->(x+y),Q)
    @test dQ ≈ StructArray{SVector{2,Float64}}((target,target))        
end