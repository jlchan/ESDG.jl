"""
connect_mesh(EToV,fv)

Initialize quad elements connectivity matrices, element to element and
element to face connectivity. Uses fv = array of indices of face vertices

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
