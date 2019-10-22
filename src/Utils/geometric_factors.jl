"""
    geometric_factors(x, y, Dr, Ds)
    geometric_factors(x, y, z, Dr, Ds, Dt)

Compute metrics of mappings between "real" elements and reference elements,
outward pointing normals on faces of every elements, and Jacobian.

Metrics in 3D are constructed to ensure satisfaction of free-stream preservation

# Examples
```jldoctest

"""
function geometric_factors(x, y, Dr, Ds)
    "Transformation and Jacobian"

    xr = Dr*x;  xs = Ds*x;
    yr = Dr*y;   ys = Ds*y;

    J = -xs.*yr + xr.*ys;
    rxJ =  ys;  sxJ = -yr;
    ryJ = -xs;  syJ =  xr;

    return rxJ, sxJ, ryJ, syJ, J
end


function geometric_factors(x, y, z, Dr, Ds, Dt)

    xr = Dr*x;  xs = Ds*x;  xt = Dt*x
    yr = Dr*y;  ys = Ds*y;  yt = Dt*y
    zr = Dr*z;  zs = Ds*z;  zt = Dt*z

    Fr = (Dr*y).*z
    Fs = (Ds*y).*z
    Ft = (Dt*y).*z
    rxJ = Dt*(Fs) - Ds*(Ft)
    sxJ = Dr*(Ft) - Dt*(Fr)
    txJ = Ds*(Fr) - Dr*(Fs)

    Fr = (Dr*x).*z
    Fs = (Ds*x).*z
    Ft = (Dt*x).*z
    ryJ = -(Dt*(Fs) - Ds*(Ft))
    syJ = -(Dr*(Ft) - Dt*(Fr))
    tyJ = -(Ds*(Fr) - Dr*(Fs))

    Fr = (Dr*y).*x
    Fs = (Ds*y).*x
    Ft = (Dt*y).*x
    rzJ = -(Dt*(Fs) - Ds*(Ft))
    szJ = -(Dr*(Ft) - Dt*(Fr))
    tzJ = -(Ds*(Fr) - Dr*(Fs))

    J = @. xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt)

    return rxJ, sxJ, txJ, ryJ, syJ, tyJ, rzJ, szJ, tzJ, J
end
