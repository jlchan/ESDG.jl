"""
function plotting_interpolation_matrix(Nplot,rd)

Computes matrix which interpolates from reference interpolation points to equispaced points of degree `Nplot`.
"""
function plotting_interpolation_matrix(Nplot,rd)
    rp,sp = NodesAndModes.equi_nodes(rd.elemShape,Nplot)
    Vp = NodesAndModes.vandermonde(rd.elemShape,rd.N,rp,sp) / rd.VDM
    return Vp
end

function triangle_area(tri)
    A,B,C = tri
    return .5*(A[1]*(B[2] - C[2]) + B[1]*(C[2]-A[2]) + C[1]*(A[2]-B[2]))
end

function plotting_triangulation(rst_plot)
    # triangulate plotting nodes on ref element (fast because thanks J. Shewchuk!)
    triin = Triangulate.TriangulateIO()
    triin.pointlist = permutedims(hcat(rst_plot...))
    triout,_ = triangulate("Q", triin)
    t = triout.trianglelist # reference element plotting triangulation

    # filter out slivers
    has_volume = fill(true,size(t,2))        
    # volume = zeros(size(t,1))
    for i = 1:size(t,2)
        ids = @view t[:,i]
        area = ESDG.triangle_area(zip(triout.pointlist[1,ids],triout.pointlist[2,ids]))
        if abs(area) < 50*eps()
            has_volume[i] = false
        end
    end 
    return t[:,findall(has_volume)]
end