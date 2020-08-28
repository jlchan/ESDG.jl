"""
connect_mesh(EToV,fv)

Initialize element connectivity matrices, element to element and
element to face connectivity. Works on general element types, so long as

        fv = array of arrays containing (unordered) indices of face vertices

is provided.

# Examples
- connect_mesh(EToV,UniformTriMesh.fv())
- connect_mesh(EToV,[[1,2],[2,3],[3,1]])
```jldoctest
```
"""
function connect_mesh(EToV,fv)
        Nfaces = length(fv)
        K = size(EToV,1)

        # sort and find matches
        fnodes = [[sort(EToV[e,ids]) for ids = fv, e = 1:K]...]
        p = sortperm(fnodes) # sorts by lexicographic ordering by default
        fnodes = fnodes[p,:]

        FToF = reshape(collect(1:Nfaces*K),Nfaces,K)
        for f = 1:size(fnodes,1)-1
                if fnodes[f,:]==fnodes[f+1,:]
                        f1 = FToF[p[f]]
                        f2 = FToF[p[f+1]]
                        FToF[p[f]] = f2
                        FToF[p[f+1]] = f1
                end
        end
        return FToF
end


"""
    function readGmsh2D(filename)

reads GMSH 2D file format 2.2 0 8
returns EToV,VX,VY

# Examples

EToV,VX,VY = readGmsh2D("eulerSquareCylinder2D.msh")

```jldoctest
"""
function readGmsh2D(filename)
    f = open(filename)
    lines = readlines(f)

    function findline(name,lines)
        for (i,line) in enumerate(lines)
            if line==name
                return i
            end
        end
    end

    node_start = findline("\$Nodes",lines)+1
    Nv = parse(Int64,lines[node_start])
    VX,VY,VZ = ntuple(x->zeros(Float64,Nv),3)
    for i = 1:Nv
        vals = [parse(Float64,c) for c in split(lines[i+node_start])]
        # first entry =
        VX[i] = vals[2]
        VY[i] = vals[3]
    end

    elem_start = findline("\$Elements",lines)+1
    K_all      = parse(Int64,lines[elem_start])
    K = 0
    for e = 1:K_all
        if length(split(lines[e+elem_start]))==8
            K = K + 1
        end
    end
    EToV = zeros(Int64,K,3)
    sk = 1
    for e = 1:K_all
        if length(split(lines[e+elem_start]))==8
            vals = [parse(Int64,c) for c in split(lines[e+elem_start])]
            EToV[sk,:] .= vals[6:8]
            sk = sk + 1
        end
    end

    EToV = EToV[:,vec([1 3 2])] # permute for Gmsh ordering

    return EToV,VX,VY
end
