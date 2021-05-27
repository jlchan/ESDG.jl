import TriplotRecipes: DGTriPseudocolor

include("vis_utils.jl")

"""
    function DGTriPseudocolor(u_plot,rd::RefElemData,md::MeshData)

Plots `u` the solution on a 2D mesh. Assumes `size(u_plot,1)==size(rd.Vp,2)`, e.g., 
`u` is evaluated at interpolation nodes. 
"""
function TriplotRecipes.DGTriPseudocolor(u,rd::RefElemData,md::MeshData)
    return TriplotRecipes.DGTriPseudocolor(rd.Vp*u,rd.rstp,(x->rd.Vp*x).(md.xyz))
end

"""
    function DGTriPseudocolor(u, Nplot::Int,rd::RefElemData,md::MeshData)

Interpolates solution `u` to a polynomial of degree `Nplot`. Assumes `size(u_plot,1)==size(rd.Vp,2)`, e.g., 
`u` is evaluated at the nodal points `md.x`, `md.y` and not plotting points.
"""
function TriplotRecipes.DGTriPseudocolor(u,Nplot::Int,rd::RefElemData,md::MeshData) 
    Vp = plotting_interpolation_matrix(Nplot,rd)
    return DGTriPseudocolor(Vp*u,(x->Vp*x).(rd.rst),(x->Vp*x).(md.xyz))
end

"""
    function TriplotRecipes.DGTriPseudocolor(u_plot,Vp::Matrix,rd::RefElemData,md::MeshData)

Returns a DGTriPseudocolor plot recipe from TriplotRecipes. 
Inputs: 
    - u_plot = matrix of size (Nplot,K) representing solution to plot. 
    - rst_plot = tuple of vector of reference plotting points of length = Nplot
    - xyz_plot = plotting points (tuple of matrices of size (Nplot,K))
"""

function TriplotRecipes.DGTriPseudocolor(u_plot, rst_plot, xyz_plot)

    @assert size(first(xyz_plot),1) == size(u_plot,1) "Row dimension of u_plot does not match row dimension of xyz_plot"
    @assert size(first(rst_plot),1) == size(u_plot,1) "Row dimension of u_plot does not match row dimension of rst_plot"

    Nplot,K = size(u_plot)

    t = plotting_triangulation(rst_plot)

    # build discontinuous data on plotting triangular mesh 
    num_ref_elements = size(t,2)
    num_elements_total = num_ref_elements * K
    tp = zeros(Int,3,num_elements_total)
    zp = similar(tp,eltype(u_plot))
    for e = 1:K
        for i = 1:size(t,2)
            tp[:,i + (e-1)*num_ref_elements] .= @views t[:,i] .+ (e-1)*Nplot
            zp[:,i + (e-1)*num_ref_elements] .= @views u_plot[t[:,i],e]
        end
    end
    return DGTriPseudocolor(vec.(xyz_plot)...,zp,tp)
end
