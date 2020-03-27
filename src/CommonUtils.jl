"""
    Module CommonUtils

General purpose utilities usable by all element types

"""

module CommonUtils
using LinearAlgebra # for I matrix in geometricFactors
using SparseArrays  # for spdiagm

export meshgrid
export geometric_factors
export connect_mesh
export build_node_maps, build_periodic_boundary_maps, build_periodic_boundary_maps!
export rk45_coeffs

export unzip # convert array of tuples to tuples of arrays
unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
export eye,speye
eye(n) = diagm(ones(n))
speye(n) = spdiagm(0 => ones(n))

# 4th order 5-stage low storage Runge Kutta from Carpenter/Kennedy.
function rk45_coeffs()
    rk4a = [            0.0 ...
    -567301805773.0/1357537059087.0 ...
    -2404267990393.0/2016746695238.0 ...
    -3550918686646.0/2091501179385.0  ...
    -1275806237668.0/842570457699.0];

    rk4b = [ 1432997174477.0/9575080441755.0 ...
    5161836677717.0/13612068292357.0 ...
    1720146321549.0/2090206949498.0  ...
    3134564353537.0/4481467310338.0  ...
    2277821191437.0/14882151754819.0]

    rk4c = [ 0.0  ...
    1432997174477.0/9575080441755.0 ...
    2526269341429.0/6820363962896.0 ...
    2006345519317.0/3224310063776.0 ...
    2802321613138.0/2924317926251.0 ...
    1.0];
    return rk4a,rk4b,rk4c
end

include("./CommonUtils/meshgrid.jl")
include("./CommonUtils/geometric_factors.jl")
include("./CommonUtils/connect_mesh.jl")
include("./CommonUtils/build_node_maps.jl")
include("./CommonUtils/build_periodic_boundary_maps.jl")

end
