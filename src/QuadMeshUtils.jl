"""
Module QuadMeshUtils

Includes mesh related utility functions
"""

module QuadMeshUtils

using SparseArrays
using Utils # for meshgrid

export uniform_quad_mesh, connect_2D

"""
uniform_quad_mesh(Kx::Int,Ky::Int)

Matlab uniform triangular mesh.

# Examples
```jldoctest
```
"""

function uniform_quad_mesh(Nx,Ny)

        Nxp = Nx+1;
        Nyp = Ny+1;
        Nv = convert(Int,Nxp*Nyp);
        K = convert(Int,Nx*Ny);

        x1D = LinRange(-1,1,Nxp);
        y1D = LinRange(-1,1,Nyp);
        x, y = meshgrid(x1D,y1D);
        I, J = meshgrid(collect(1:Nxp),collect(1:Nyp));
        inds = @. (I-1)*Ny + (J+I-1);
        EToV = zeros(Int,K,4);
        k = 1;
        for i = 1:Ny
                for j = 1:Nx
                        EToV[k,:] = [inds[i,j] inds[i,j+1] inds[i+1,j+1] inds[i+1,j]];
                        k += 1;
                end
        end

        VX = x[:];
        VY = y[:];

        return VX[:],VY[:],EToV
end

function uniform_quad_mesh(Kx)
        return uniform_quad_mesh(Kx,Ky)
end

"""
connect_2D(EToV)

Initialize quad elements connectivity matrices, element to element and
element to face connectivity.

# Examples
```jldoctest
"""
function connect_2D(EToV)
        Nfaces = 4
        K = size(EToV,1)
        Nnodes = maximum(EToV)
        fnodes = [EToV[:,[1,2]]; EToV[:,[2,3]]; EToV[:,[3,4]]; EToV[:,[4,1]]]
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
