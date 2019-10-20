"""
Module TriMeshUtils

Includes mesh related utility functions
"""

module TriMeshUtils

using SparseArrays
using Utils # for meshgrid

export uniform_tri_mesh, connect_2D


"""
uniform_tri_mesh(Kx::Int,Ky::Int)

Matlab uniform triangular mesh.

# Examples
```jldoctest
```
"""

function uniform_tri_mesh(Kx,Ky)
        (VY, VX) = meshgrid(LinRange(-1,1,Kx+1),LinRange(-1,1,Ky+1))
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
        return uniform_tri_mesh(Kx,Ky)
end

"""
connect_2D(Nfaces, K, EToV)

Initialize triangle elements connectivity matrices, element to element and
element to face connectivity.

# Examples
```jldoctest
"""
function connect_2D(EToV)
        Nfaces = 3 # assumed since it's a triangular routine
        K = size(EToV,1)
        Nnodes = maximum(EToV)
        fnodes = [EToV[:,[1,2]]; EToV[:,[2,3]]; EToV[:,[3,1]]]
        sort!(fnodes, dims = 2)
        fnodes = fnodes.-1;
        EToE = (1:K)*ones(Int64,1,Nfaces)
        EToF = ones(Int64,K,1)*transpose(1:Nfaces)
        id = fnodes[:,1]*Nnodes + fnodes[:,2].+1;
        spNodeToNode = [id collect(1:Nfaces*K) EToE[:] EToF[:]]
        sorted = sortslices(spNodeToNode, dims=1)
        indices = findall(sorted[1:(end-1),1] .== sorted[2:end,1])
        matchL = [sorted[indices,:]; sorted[indices.+1,:]]
        matchR = [sorted[indices.+1,:]; sorted[indices,:]]
        EToE[matchL[:,2]] = matchR[:,3]
        EToF[matchL[:,2]] = matchR[:,4]
        return EToE, EToF
end



end
