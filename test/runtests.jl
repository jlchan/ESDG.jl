using ESDG
using Test

@testset "Test accuracy on unstructured tri meshes" begin
    triout = square_hole_domain()
    VX,VY,EToV = triangulateIO_to_VXYEToV(triout)
    rd = RefElemData(Tri(),N=10)
    md = MeshData(VX,VY,EToV,rd)

    u_exact(x,y) = sin(pi*x)*sin(pi*y)
    dudx_exact(x,y) = pi*cos(pi*x)*sin(pi*y)

    @unpack x,y,rxJ,sxJ,J = md
    u = u_exact.(x,y)
    dudx = (rxJ.*(rd.Dr*u) + sxJ.*(rd.Ds*u))./J
    @test dudx ≈ dudx_exact.(x,y)
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