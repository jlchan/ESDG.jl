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
        Nnodes = maximum(EToV)

        # sort and find matches
        fnodes = [[sort(EToV[e,ids]) for ids = fv, e = 1:K]...]
        p = sortperm(fnodes)
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

function connect_mesh_old(EToV,fv)
        Nfaces = length(fv)
        K = size(EToV,1)
        Nnodes = maximum(EToV)

        fnodes = vcat([EToV[:,ids] for ids = fv]...) # vertically concat array comprehension
        sort!(fnodes, dims = 2) # sort rows to be ascending

        # @show fnodes
        fnodes = fnodes .- 1;
        EToE = (1:K)*ones(Int64,1,Nfaces)
        EToF = ones(Int64,K,1)*transpose(1:Nfaces)

        # compute ids - may overflow around 50000 nodes
        if Nnodes > 50000
                error("too many nodes! may result in overflow")
        end
        # give "nicknames" - should replace with an ordering operator
        Nfv = size(fnodes,2)
        id = zeros(size(fnodes,1))
        for i = 1:Nfv
                id .+= fnodes[:,i].*Nnodes^(Nfv-i+1)
        end
        id = convert.(Int,round.(id .+ 1))
        # id = fnodes[:,1]*Nnodes + fnodes[:,2] .+ 1;

        spNodeToNode = [id collect(1:Nfaces*K) EToE[:] EToF[:]]
        sorted = sortslices(spNodeToNode, dims=1)
        indices = findall(sorted[1:(end-1),1] .== sorted[2:end,1])
        matchL = [sorted[indices,:]; sorted[indices.+1,:]]
        matchR = [sorted[indices.+1,:]; sorted[indices,:]]
        EToE[matchL[:,2]] = matchR[:,3]
        EToF[matchL[:,2]] = matchR[:,4]

        FToF = reshape(collect(1:Nfaces*K),Nfaces,K)
        for e = 1:K
            for f = 1:Nfaces
                FToF[f,e] = EToF[e,f] + (EToE[e,f]-1)*Nfaces
            end
        end

        return EToE, EToF, FToF
end
