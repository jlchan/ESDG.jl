"""
Module UniformQuadMesh

Includes mesh related utility functions
"""

module UniformQuadMesh

using SparseArrays
using CommonUtils # for meshgrid

export uniform_quad_mesh
export quad_face_vertices

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
                        EToV[k,:] = [inds[i,j] inds[i,j+1] inds[i+1,j] inds[i+1,j+1]];
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

# ordering matters...
function quad_face_vertices()
        return [1,2],[2,4],[3,4],[1,3]
end


end
