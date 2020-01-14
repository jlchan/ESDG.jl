"""
Module UniformHexMesh

Includes mesh related utility functions
"""

module UniformHexMesh

using SparseArrays
using CommonUtils # for meshgrid

export uniform_hex_mesh
export hex_face_vertices

"""
uniform_hex_mesh(Kx::Int,Ky::Int)

Matlab uniform hexahedral mesh.

# Examples
```jldoctest
```
"""

function uniform_hex_mesh(Nx,Ny,Nz)
        Nxp = Nx+1
        Nyp = Ny+1
        Nzp = Nz+1
        Nv = convert(Int,Nxp*Nyp*Nzp)
        K = convert(Int,Nx*Ny*Nz)

        x1D = LinRange(-1,1,Nxp)
        y1D = LinRange(-1,1,Nyp)
        z1D = LinRange(-1,1,Nzp)
        x, y, z = meshgrid(x1D,y1D,z1D)

        EToV = zeros(Int,K,8)
        k = 1
        for e = 1:K
                em = e-1
                k = div(em,(Nx*Ny))
                j = div(em - k*Nx*Ny,Nx)
                i = em % Nx

                EToV[e,1] = i     + Nxp*j     + Nxp*Nyp*k
                EToV[e,2] = (i+1) + Nxp*j     + Nxp*Nyp*k
                EToV[e,3] = i     + Nxp*(j+1) + Nxp*Nyp*k
                EToV[e,4] = (i+1) + Nxp*(j+1) + Nxp*Nyp*k
                EToV[e,5] = i     + Nxp*j     + Nxp*Nyp*(k+1)
                EToV[e,6] = (i+1) + Nxp*j     + Nxp*Nyp*(k+1)
                EToV[e,7] = i     + Nxp*(j+1) + Nxp*Nyp*(k+1)
                EToV[e,8] = (i+1) + Nxp*(j+1) + Nxp*Nyp*(k+1)
        end
        EToV = @. EToV + 1 # re-index to 1 index

        VX = x[:];
        VY = y[:];
        VZ = z[:];
        return VX[:],VY[:],VZ[:],EToV
end

function uniform_hex_mesh(Kx)
        return uniform_hex_mesh(Kx,Kx,Kx)
end


function hex_face_vertices()
        x1D = LinRange(-1,1,2)
        r, s, t = meshgrid(x1D,x1D,x1D)
        fv1 = map(x->x[1], findall(@. abs(r+1) < 1e-10))
        fv2 = map(x->x[1], findall(@. abs(r-1) < 1e-10))
        fv3 = map(x->x[1], findall(@. abs(s+1) < 1e-10))
        fv4 = map(x->x[1], findall(@. abs(s-1) < 1e-10))
        fv5 = map(x->x[1], findall(@. abs(t+1) < 1e-10))
        fv6 = map(x->x[1], findall(@. abs(t-1) < 1e-10))
        return (fv1,fv2,fv3,fv4,fv5,fv6)
end


end
