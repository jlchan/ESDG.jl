using ESDG, Test
using LinearAlgebra
using StaticArrays
using StructArrays

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

@testset "SBP tests" begin
    @testset "DiagE operators" begin
        N = 3
        for quadrature_strength = [2*N-1,2*N] # 2*N-1, 2*N quad strength
            sbp,rd = DiagESummationByParts(Tri(),N,quadrature_strength=quadrature_strength);
            Qr,Qs = sbp.Qrst
            @test Qr+Qr' ≈ sbp.Ef'*diagm(rd.nrJ .* rd.wf)*sbp.Ef
            @test Qs+Qs' ≈ sbp.Ef'*diagm(rd.nsJ .* rd.wf)*sbp.Ef
            r,s = sbp.points
            @test ((Qr*r.^N)./sbp.wq ≈ N*r.^(N-1)) && ((Qs*s.^N)./sbp.wq ≈ N*s.^(N-1))
            @test (norm(sum(Qr,dims=2)) < 50*eps()) && (norm(sum(Qs,dims=2)) < 50*eps())
        end
    end
    @testset "Hybridized operators" begin
        N = 3
        rd = RefElemData(Tri(),N)
        Qrh,Qsh,VhP,Ph = hybridized_SBP_operators(rd)
        @unpack r,rq,rf = rd
        rh = [rq;rf]        

        @test norm(sum(Qrh,dims=2)) + norm(sum(Qsh,dims=2)) < 50*eps()
        @test N*r.^(N-1) ≈ Ph*Qrh*(rh.^N)
    end
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