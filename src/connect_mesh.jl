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
