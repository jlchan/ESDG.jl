"""
Module UniformTriMesh

Includes mesh related utility functions
"""

module UniformTriMesh

using SparseArrays
using CommonUtils # for meshgrid

export uniform_tri_mesh
export tri_face_vertices

"""
uniform_tri_mesh(Kx::Int,Ky::Int)

Matlab uniform triangular mesh.

# Examples
```jldoctest
```
"""

function uniform_tri_mesh(Kx,Ky)
        (VY, VX) = meshgrid(LinRange(-1,1,Ky+1),LinRange(-1,1,Kx+1))
        sk = 1
        EToV = zeros(Int,2*Kx*Ky,3)
        for ey = 1:Ky
                for ex = 1:Kx
                        id(ex,ey) = ex + (ey-1)*(Kx+1) # index function
                        id1 = id(ex,ey);
                        id2 = id(ex+1,ey);
                        id3 = id(ex+1,ey+1);
                        id4 = id(ex,ey+1);
                        EToV[2*sk-1,:] = [id1 id2 id3];
                        EToV[2*sk,:] = [id3 id4 id1];
                        sk += 1
                end
        end
        return (VX[:],VY[:],EToV)
end

function uniform_tri_mesh(Kx)
        return uniform_tri_mesh(Kx,Kx)
end


function tri_face_vertices()
        return [1,2],[2,3],[3,1]
end


end
