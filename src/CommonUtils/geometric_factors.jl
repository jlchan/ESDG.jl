"""
    geometric_factors(x, y, Dr, Ds)
    geometric_factors(x, y, z, Dr, Ds, Dt)

Compute metrics of mappings between "real" elements and reference elements,
outward pointing normals on faces of every elements, and Jacobian.

x,y,z are arrays of coordinates, and Dr, Ds, Dt are nodal differentiation matrices

Geometric terms in 3D are constructed to ensure satisfaction of free-stream
preservation using the curl-based construction of David Kopriva (2001).

"""

# 2D version
function geometric_factors(x, y, Dr, Ds)
    "Transformation and Jacobian"

    xr = Dr*x;   xs = Ds*x;
    yr = Dr*y;   ys = Ds*y;

    J = -xs.*yr + xr.*ys;
    rxJ =  ys;  sxJ = -yr;
    ryJ = -xs;  syJ =  xr;

    return rxJ, sxJ, ryJ, syJ, J
end
