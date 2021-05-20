
@testset "Test premade meshes" for premade_mesh in [square_hole_domain, rectangular_domain]
    triout = premade_mesh()
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