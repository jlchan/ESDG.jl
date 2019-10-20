"""
    meshgrid(x, y)

Matlab meshgrid function. Return 2D grid coordinates according to x and y.

# Examples
```jldoctest
julia> R = meshgrid(1:3, 1:3)
([1 2 3; 1 2 3; 1 2 3], [1 1 1; 2 2 2; 3 3 3])
```
"""
function meshgrid(x,y)
    X = repeat(transpose(x[:]), size(y[:], 1), 1)
    Y = repeat(y, 1, size(x, 1))
    return X, Y
end

"""
    meshgrid(x)

Matlab meshgrid function. Return 2D grid coordinates according to x and y=x.

# Examples
```jldoctest
```
"""
function meshgrid(x)
    return meshgrid(x,x)
end
