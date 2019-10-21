"""
connect_mesh(EToV,fv)

Initialize quad elements connectivity matrices, element to element and
element to face connectivity. uses fv = array of ids of face nodes

# Examples
```jldoctest
"""
function connect_mesh(EToV,fv)
        Nfaces = length(fv)
        K = size(EToV,1)
        Nnodes = maximum(EToV)
        fnodes = vcat([EToV[:,ids] for ids = fv]...) # vertically concat array comprehension

        sort!(fnodes, dims = 2)
        fnodes = fnodes.-1;
        EToE = (1:K)*ones(Int64,1,Nfaces)
        EToF = ones(Int64,K,1)*transpose(1:Nfaces)
        id = fnodes[:,1]*Nnodes + fnodes[:,2].+1;
        spNodeToNode = [id collect(1:Nfaces*K) EToE[:] EToF[:]]
        sorted = sortslices(spNodeToNode, dims=1)
        indices = findall(sorted[1:(end-1),1] .== sorted[2:end,1])
        matchL = [sorted[indices,:]; sorted[indices.+1,:]]
        matchR = [sorted[indices.+1,:]; sorted[indices,:]]
        EToE[matchL[:,2]] = matchR[:,3]
        EToF[matchL[:,2]] = matchR[:,4]

        return EToE, EToF
end
