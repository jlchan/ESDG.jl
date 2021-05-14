"""
    struct NodalESDG{ElemType,DIM,F1,F2,F3,Tv,Ti}
        sbp_operators::DiagESummationByParts{ElemType,DIM,Tv,Ti} # non-polynomial SBP operators
        rd::RefElemData{DIM,ElemType}
        volume_flux::F1 
        interface_flux::F2
        interface_dissipation::F3
    end
Entropy stable solver using nodal (collocated) DG methods
"""
struct NodalESDG{ElemType,DIM,F1,F2,F3,Tv,Ti}
    sbp_operators::DiagESummationByParts{ElemType,DIM,Tv,Ti} # non-polynomial SBP operators
    rd::RefElemData{DIM,ElemType,Tv} 
    volume_flux::F1 
    interface_flux::F2
    interface_dissipation::F3
end

function Base.show(io::IO, solver::NodalESDG{ElemType,DIM}) where {ElemType,DIM}
    println("Nodal ESDG solver for element type $ElemType in $DIM dimension with ")
    println("   volume flux           = $(solver.volume_flux.trixi_volume_flux)")
    println("   interface flux        = $(solver.interface_flux.trixi_interface_flux)")    
    println("   interface dissipation = $(solver.interface_dissipation.trixi_interface_dissipation)")        
end

function NodalESDG(N,elementType,
                   trixi_volume_flux::F1,
                   trixi_interface_flux::F2,
                   trixi_interface_dissipation::F3,
                   equations; quadrature_strength=2*N-1) where {F1,F2,F3}
    
    volume_flux, interface_flux, interface_dissipation = let equations=equations
        volume_flux(orientation) = (u_ll,u_rr)->trixi_volume_flux(u_ll,u_rr,orientation,equations)
        interface_flux(orientation) = (u_ll,u_rr)->trixi_interface_flux(u_ll,u_rr,orientation,equations)
        interface_dissipation(orientation) = (u_ll,u_rr)->trixi_interface_dissipation(u_ll,u_rr,orientation,equations)
        volume_flux,interface_flux,interface_dissipation
    end

    quad_rule_vol,quad_rule_face = diagE_sbp_nodes(elementType, N; quadrature_strength=quadrature_strength)
    rd_sbp = RefElemData(elementType,N; quad_rule_vol = quad_rule_vol, quad_rule_face = quad_rule_face)
    sbp_ops = DiagESummationByParts(elementType, N, rd_sbp)

    NodalESDG(sbp_ops,rd_sbp,volume_flux,interface_flux,interface_dissipation)
end

