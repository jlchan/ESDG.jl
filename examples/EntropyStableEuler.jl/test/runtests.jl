using EntropyStableEuler
using Test

@testset "Logmean tests" begin
    uL,uR = 1,2
    @test logmean(uL,uR) == logmean(uL,uR,log(uL),log(uR))
    @test logmean(uL,uR) == logmean(uR,uL)
    @test logmean(uL,uL) ≈ uL
end


@testset "1D entropy variable tests" begin
    using EntropyStableEuler.Fluxes1D
    import EntropyStableEuler: γ,entropy_scaling

    rho,u,p = 1,.1,2
    rho,rhou,E = Fluxes1D.primitive_to_conservative(rho,u,p)
    v1,v2,v3 = Fluxes1D.v_ufun(rho,rhou,E)

    h = 1e-7
    central_diff(f,x) = (f(x+h) - f(x-h))/(2*h)
    @test abs(v1 - central_diff(rho->Fluxes1D.Sfun(rho,rhou,E),rho)) < h
    @test abs(v2 - central_diff(rhou->Fluxes1D.Sfun(rho,rhou,E),rhou)) < h
    @test abs(v3 - central_diff(E->Fluxes1D.Sfun(rho,rhou,E),E)) < h

    u1,u2,u3 = Fluxes1D.u_vfun(v1,v2,v3)
    @test u1 ≈ rho
    @test u2 ≈ rhou
    @test u3 ≈ E

    # test symmetry
    UL = copy.((rho,rhou,E))
    VL = copy.((v1,v2,v3))
    UR = Fluxes1D.primitive_to_conservative(1.1,.2,2.1)
    VR = Fluxes1D.v_ufun(UR...)
    QL = Fluxes1D.conservative_to_primitive_beta(UL...)
    QR = Fluxes1D.conservative_to_primitive_beta(UR...)
    Fx = Fluxes1D.euler_fluxes(QL...,QR...)
    Fx2 = Fluxes1D.euler_fluxes(QR...,QL...)
    @test all(Fx .≈ Fx2)

    # test consistency
    p = Fluxes1D.pfun(rho,rhou,E)
    exact_flux_x = (rho*u, rho*u^2 + p, u*(E+p))
    FxL = Fluxes1D.euler_fluxes(QL...,QL...)
    @test all(FxL .≈ exact_flux_x)

    # test entropy conservation property
    # entropy potentials
    ψx(U) = (γ-1)*U[2]*entropy_scaling
    vTFx = sum(((x,y,z)->((x-y)*z)).(VL,VR,Fx))
    @test vTFx ≈ ψx(UL)-ψx(UR)
end

@testset "2D entropy variable tests" begin
    using EntropyStableEuler.Fluxes2D
    import EntropyStableEuler: γ,entropy_scaling

    rho,u,v,p = 1,.1,.2,2
    rho,rhou,rhov,E = Fluxes2D.primitive_to_conservative(rho,u,v,p)
    v1,v2,v3,v4 = Fluxes2D.v_ufun(rho,rhou,rhov,E)

    h = 1e-7
    central_diff(f,x) = (f(x+h) - f(x-h))/(2*h)

    @test abs(v1 - central_diff(rho->Fluxes2D.Sfun(rho,rhou,rhov,E),rho)) < h
    @test abs(v2 - central_diff(rhou->Fluxes2D.Sfun(rho,rhou,rhov,E),rhou)) < h
    @test abs(v3 - central_diff(rhov->Fluxes2D.Sfun(rho,rhou,rhov,E),rhov)) < h
    @test abs(v4 - central_diff(E->Fluxes2D.Sfun(rho,rhou,rhov,E),E)) < h

    u1,u2,u3,u4 = Fluxes2D.u_vfun(v1,v2,v3,v4)
    @test u1 ≈ rho
    @test u2 ≈ rhou
    @test u3 ≈ rhov
    @test u4 ≈ E

    # test symmetry
    UL = copy.((rho,rhou,rhov,E))
    VL = copy.((v1,v2,v3,v4))
    UR = Fluxes2D.primitive_to_conservative(1.1,.2,.3,2.1)
    VR = Fluxes2D.v_ufun(UR...)
    QL = Fluxes2D.conservative_to_primitive_beta(UL...)
    QR = Fluxes2D.conservative_to_primitive_beta(UR...)
    Fx,Fy = Fluxes2D.euler_fluxes(QL...,QR...)
    Fx2,Fy2 = Fluxes2D.euler_fluxes(QR...,QL...)
    @test all(Fx .≈ Fx2)
    @test all(Fy .≈ Fy2)

    # test consistency
    p = Fluxes2D.pfun(rho,rhou,rhov,E)
    exact_flux_x = (rho*u, rho*u^2 + p, rhou*v, u*(E+p))
    exact_flux_y = (rho*v, rhou*v, rho*v^2 + p, v*(E+p))
    FxL,FyL = Fluxes2D.euler_fluxes(QL...,QL...)
    @test all(FxL .≈ exact_flux_x)
    @test all(FyL .≈ exact_flux_y)

    # test individual coordinate fluxes
    @test all(Fx .≈ Fluxes2D.euler_fluxes_2D_x(QL...,QR...))
    @test all(Fy .≈ Fluxes2D.euler_fluxes_2D_y(QL...,QR...))

    # test entropy conservation property
    # entropy potentials
    ψx(U) = (γ-1)*U[2]*entropy_scaling
    ψy(U) = (γ-1)*U[3]*entropy_scaling
    vTFx = sum(((x,y,z)->((x-y)*z)).(VL,VR,Fx))
    vTFy = sum(((x,y,z)->((x-y)*z)).(VL,VR,Fy))
    @test vTFx ≈ ψx(UL)-ψx(UR)
    @test vTFy ≈ ψy(UL)-ψy(UR)

    # test type stability
    @inferred Fluxes2D.primitive_to_conservative(1.1,.2,.3,2.1)
    @inferred Fluxes2D.v_ufun(UR...)
    @inferred Fluxes2D.conservative_to_primitive_beta(UL...)
    @inferred Fluxes2D.conservative_to_primitive_beta(UR...)
    @inferred Fluxes2D.euler_fluxes(QL...,QR...)
    # using StaticArrays
    # Q = rho,u,v,p
    # U = rho,rhou,rhov,E
    # VU = v1,v2,v3,v4
    # Q = SVector(Q)
    # U = SVector(U)
    # VU = SVector(VU)
    # @inferred Fluxes2D.primitive_to_conservative(Q...);
    # @inferred Fluxes2D.v_ufun(U...);
    # @inferred Fluxes2D.u_vfun(VU...);
    # @inferred Fluxes2D.conservative_to_primitive_beta(U...);
end


@testset "3D entropy variable tests" begin
    using EntropyStableEuler.Fluxes3D
    import EntropyStableEuler: γ,entropy_scaling

    rho,u,v,w,p = 1,.1,.2,.3,2
    rho,rhou,rhov,rhow,E = Fluxes3D.primitive_to_conservative(rho,u,v,w,p)
    v1,v2,v3,v4,v5 = Fluxes3D.v_ufun(rho,rhou,rhov,rhow,E)

    h = 1e-7
    central_diff(f,x) = (f(x+h) - f(x-h))/(2*h)
    @test abs(v1 - central_diff(rho->Fluxes3D.Sfun(rho,rhou,rhov,rhow,E),rho)) < h
    @test abs(v2 - central_diff(rhou->Fluxes3D.Sfun(rho,rhou,rhov,rhow,E),rhou)) < h
    @test abs(v3 - central_diff(rhov->Fluxes3D.Sfun(rho,rhou,rhov,rhow,E),rhov)) < h
    @test abs(v4 - central_diff(rhow->Fluxes3D.Sfun(rho,rhou,rhov,rhow,E),rhow)) < h
    @test abs(v5 - central_diff(E->Fluxes3D.Sfun(rho,rhou,rhov,rhow,E),E)) < h

    u1,u2,u3,u4,u5 = Fluxes3D.u_vfun(v1,v2,v3,v4,v5)
    @test u1 ≈ rho
    @test u2 ≈ rhou
    @test u3 ≈ rhov
    @test u4 ≈ rhow
    @test u5 ≈ E

    # test symmetry
    UL = copy.((rho,rhou,rhov,rhow,E))
    VL = copy.((v1,v2,v3,v4,v5))
    UR = Fluxes3D.primitive_to_conservative(1.1,.2,.3,.4,2.1)
    VR = Fluxes3D.v_ufun(UR...)
    QL = Fluxes3D.conservative_to_primitive_beta(UL...)
    QR = Fluxes3D.conservative_to_primitive_beta(UR...)
    Fx,Fy,Fz = Fluxes3D.euler_fluxes(QL...,QR...)
    Fx2,Fy2,Fz2 = Fluxes3D.euler_fluxes(QR...,QL...)
    @test all(Fx .≈ Fx2)
    @test all(Fy .≈ Fy2)
    @test all(Fz .≈ Fz2)

    # test consistency
    p = Fluxes3D.pfun(rho,rhou,rhov,rhow,E)
    exact_flux_x = (rho*u, rho*u^2 + p, rhou*v,      rhou*w,      u*(E+p))
    exact_flux_y = (rho*v, rhov*u,      rho*v^2 + p, rhov*w,      v*(E+p))
    exact_flux_z = (rho*w, rhow*u,      rhow*v,      rho*w^2 + p, w*(E+p))
    FxL,FyL,FzL = Fluxes3D.euler_fluxes(QL...,QL...)
    @test all(FxL .≈ exact_flux_x)
    @test all(FyL .≈ exact_flux_y)
    @test all(FzL .≈ exact_flux_z)

    # test type stability
    @inferred Fluxes3D.primitive_to_conservative(1.1,.2,.3,.4,2.1)
    @inferred Fluxes3D.v_ufun(UR...)
    @inferred Fluxes3D.conservative_to_primitive_beta(UL...)
    @inferred Fluxes3D.conservative_to_primitive_beta(UR...)
    @inferred Fluxes3D.euler_fluxes(QL...,QR...)

    # test entropy conservation property
    # entropy potentials
    ψx(U) = (γ-1)*U[2]*entropy_scaling
    ψy(U) = (γ-1)*U[3]*entropy_scaling
    ψz(U) = (γ-1)*U[4]*entropy_scaling
    vTFx = sum(((x,y,z)->((x-y)*z)).(VL,VR,Fx))
    vTFy = sum(((x,y,z)->((x-y)*z)).(VL,VR,Fy))
    vTFz = sum(((x,y,z)->((x-y)*z)).(VL,VR,Fz))
    @test vTFx ≈ ψx(UL)-ψx(UR)
    @test vTFy ≈ ψy(UL)-ψy(UR)
    @test vTFz ≈ ψz(UL)-ψz(UR)
end

# module bar
# export f
# f(x) = f(x...) # dispatch
# module foo1
# import ..bar: f
# f(x1,x2) = x1+x2
# end
# module foo2
# import ..bar: f
# f(x1,x2,x3) = x1+x2+x3
# end
# end
