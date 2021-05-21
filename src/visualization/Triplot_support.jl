import TriplotRecipes: DGTriPseudocolor

"""
function plotting_interpolation_matrix(Nplot,rd)

Computes matrix which interpolates from reference interpolation points to equispaced points of degree `Nplot`.
"""
function plotting_interpolation_matrix(Nplot,rd)
    rp,sp = NodesAndModes.equi_nodes(rd.elemShape,Nplot)
    Vp = NodesAndModes.vandermonde(rd.elemShape,rd.N,rp,sp) / rd.VDM
    return Vp
end

"""
    function DGTriPseudocolor(u_plot,rd::RefElemData,md::MeshData)

Plots `u_plot` the solution on a triangular mesh. Assumes `size(u_plot,1)==size(rd.Vp,1)`, e.g., 
`u_plot` is evaluated at the plotting nodes already. 
"""
TriplotRecipes.DGTriPseudocolor(u_plot,rd::RefElemData,md::MeshData) = DGTriPseudocolor(u_plot,rd.Vp,rd,md)

"""
    function DGTriPseudocolor(u, Nplot::Int,rd::RefElemData,md::MeshData)

Interpolates solution `u` to a polynomial of degree `Nplot`. Assumes `size(u_plot,1)==size(rd.Vp,2)`, e.g., 
`u` is evaluated at the nodal points `md.x`, `md.y` and not plotting points.
"""
function TriplotRecipes.DGTriPseudocolor(u,Nplot::Int,rd::RefElemData,md::MeshData) 
    Vp = plotting_interpolation_matrix(Nplot,rd)
    return DGTriPseudocolor(Vp*u,Vp,rd,md)
end

"""
    function TriplotRecipes.DGTriPseudocolor(u_plot,Vp::Matrix,rd::RefElemData,md::MeshData)

Returns a DGTriPseudocolor plot recipe from TriplotRecipes. 
Inputs: 
    - u_plot = matrix of size (Nplot,K) representing solution to plot. 
    - Vp = interpolation matrix of size (Nplot,Np). 
"""
function TriplotRecipes.DGTriPseudocolor(u_plot,Vp::Matrix,rd::RefElemData,md::MeshData)

    @assert size(Vp,1) == size(u_plot,1) "Row dimension of u_plot does not match row dimension of Vp"

    # plotting nodes implicitly defined by Vp and rd::RefElemData
    rp,sp = Vp*rd.r, Vp*rd.s

    # triangulate plotting nodes on ref element (fast because thanks J. Shewchuk!)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims([rp sp]) 
    triout_plot,_ = triangulate("Q", triin)
    t = triout_plot.trianglelist # reference element plotting triangulation

    # build discontinuous data on plotting triangular mesh 
    num_ref_elements = size(t,2)
    num_elements_total = num_ref_elements * md.K
    tp = zeros(Int,3,num_elements_total)
    zp = similar(tp,eltype(u_plot))
    for e = 1:md.K
        for i = 1:size(t,2)
            tp[:,i + (e-1)*num_ref_elements] .= @views t[:,i] .+ (e-1)*size(Vp,1)
            zp[:,i + (e-1)*num_ref_elements] .= @views u_plot[t[:,i],e]
        end
    end
    return DGTriPseudocolor(vec(Vp*md.x),vec(Vp*md.y),zp,tp)
end
