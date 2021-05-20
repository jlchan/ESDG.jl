using ESDG
using StaticArrays, StructArrays

n = 10
u = collect(LinRange(-1,1,n))
du = zero(u)
A = Matrix{Float64}(reshape(1:n*n,n,n))
hadamard_sum_ATr!(du,A',(x,y)->(x+y),u)
target = A*u + vec(u.*sum(A,dims=2))
@test du ≈ target

# test structarrays    
v = copy(u)
dv = similar(v)
Q = StructArray{SVector{2,Float64}}((u,v))
targetQ = StructArray{SVector{2,Float64}}((target,target))
dQ = zero(Q)
hadamard_sum_ATr!(dQ,A',(x,y)->(x+y),Q)
@test dQ ≈ targetQ

# test list of arrays + structarrays    
fill!(dQ,zero(eltype(dQ)))
hadamard_sum_ATr!(dQ,(A',A'),(x,y)->((x+y),(x+y)),Q)
@test dQ ≈ 2 .* targetQ