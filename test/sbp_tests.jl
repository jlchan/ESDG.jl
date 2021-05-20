
@testset "Triangular DiagE operators" begin
    N = 3
    for quadrature_strength = [2*N-1,2*N] # 2*N-1, 2*N quad strength
        sbp = DiagESummationByParts(Tri(),N,quadrature_strength=quadrature_strength);
        Qr,Qs = sbp.Qrst
        nrJ,nsJ = sbp.nrstJ
        wf = sbp.wf
        @test Qr+Qr' ≈ sbp.Ef'*diagm(nrJ .* wf)*sbp.Ef
        @test Qs+Qs' ≈ sbp.Ef'*diagm(nsJ .* wf)*sbp.Ef
        r,s = sbp.points
        @test ((Qr*r.^N)./sbp.wq ≈ N*r.^(N-1)) && ((Qs*s.^N)./sbp.wq ≈ N*s.^(N-1))
        @test (norm(sum(Qr,dims=2)) < 50*eps()) && (norm(sum(Qs,dims=2)) < 50*eps())
    end
end    

@testset "Quad DiagE (DGSEM) operators" begin
    N = 3
    sbp = DiagESummationByParts(Quad(),N);
    Qr,Qs = sbp.Qrst
    nrJ,nsJ = sbp.nrstJ
    wf = sbp.wf
    @test Qr+Qr' ≈ sbp.Ef'*diagm(nrJ .* wf)*sbp.Ef
    @test Qs+Qs' ≈ sbp.Ef'*diagm(nsJ .* wf)*sbp.Ef
    r,s = sbp.points
    @test ((Qr*r.^N)./sbp.wq ≈ N*r.^(N-1)) && ((Qs*s.^N)./sbp.wq ≈ N*s.^(N-1))
    @test (norm(sum(Qr,dims=2)) < 50*eps()) && (norm(sum(Qs,dims=2)) < 50*eps())
end   
     
@testset "Hybridized SBP operators" begin
    N = 3
    rd = RefElemData(Tri(),N)
    Qrh,Qsh,VhP,Ph = hybridized_SBP_operators(rd)
    @unpack r,rq,rf = rd
    rh = [rq;rf]        

    @test norm(sum(Qrh,dims=2)) + norm(sum(Qsh,dims=2)) < 50*eps()
    @test N*r.^(N-1) ≈ Ph*Qrh*(rh.^N)
end