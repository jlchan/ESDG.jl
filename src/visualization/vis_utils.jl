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
function compute_triangle_area(tri)

Computes the area of a triangle given a tuple of three points `tri`. 
Formula from https://en.wikipedia.org/wiki/Shoelace_formula
"""
function compute_triangle_area(tri)
    A,B,C = tri
    return .5*(A[1]*(B[2] - C[2]) + B[1]*(C[2]-A[2]) + C[1]*(A[2]-B[2]))
end

"""
function plotting_triangulation(rst_plot)

Computes a triangulation of the points in `rst_plot`, which is a tuple containing 
vectors of plotting points on the reference element. Returns a `3 x N_tri` matrix containing
a triangulation of the plotting points, with zero-volume triangles removed.
"""
function plotting_triangulation(rst_plot,tol=50*eps())

    # on-the-fly triangulation of plotting nodes on the reference element
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims(hcat(rst_plot...))
    triout,_ = triangulate("Q", triin)
    t = triout.trianglelist

    # filter out sliver triangles
    has_volume = fill(true,size(t,2))
    for i in axes(t,2)
        ids = @view t[:,i]
        x_points = @view triout.pointlist[1,ids]
        y_points = @view triout.pointlist[2,ids]
        area = compute_triangle_area(zip(x_points,y_points))
        if abs(area) < tol
            has_volume[i] = false
        end
    end
    return t[:,findall(has_volume)]
end