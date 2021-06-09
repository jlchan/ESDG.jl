"""
    function hadamard_sum_ATr!(rhs, ATr_list::NTuple{N}, F, u, skip_index=(i,j)->false) where {N}

Given a tuple of operators `ATr_list` and a flux `F(UL,UR)` which returns a tuple of outputs, this will 
accumulate ∑_i `sum(A_i.*F(u,u')[i],dims=2)` into `rhs`.
"""
@inline function hadamard_sum_ATr!(rhs, ATr_list::NTuple{N}, F, u, skip_index=(i,j)->false) where {N}
    rows,cols = axes(first(ATr_list))
    for i in cols
        ui = u[i]
        val_i = rhs[i]
        for j in rows
            if !skip_index(i,j)
                val_i += sum(getindex.(ATr_list,j,i) .* F(ui,u[j]))
            end
        end
        rhs[i] = val_i 
    end
end

"""
    function hadamard_sum_ATr!(rhs, ATr, F, u, skip_index=(i,j)->false)

Computes and accumulates `sum(A.*F(u,u'),dims=2)` into `rhs`. 
""" 
@inline function hadamard_sum_ATr!(rhs, ATr, F, u, skip_index=(i,j)->false)
    hadamard_sum_ATr!(rhs, (ATr,), F ∘ tuple, u, skip_index)
end

"""
    function hadamard_sum_ATr!(rhs, ATr_list::NTuple{Dim,<:AbstractSparseMatrix}, F, u)    
    function hadamard_sum_ATr!(rhs, ATr_list::NTuple{Dim,<:AbstractSparseMatrix}, ScalingMatrices, F, u)

Accumulates sum(A.*F(u,u'),dims=2) into rhs but takes advantage of sparsity of A. 

In practice, we often wish to compute `sum(A_i .* G_i .* F(u,u'),dims=2)` where `G_i` are scaling terms.
The argument `ScalingMatrices` is a tuple of matrices containing the matrices `G_i` which scale `A_i` entrywise. 
Suggested use: define ScalingMatrices as a tuple of LazyArrays for efficiency. 
"""
@inline function hadamard_sum_ATr!(rhs, ATr_list::NTuple{Dim,<:AbstractSparseMatrix}, ScalingMatrices, F, u) where {Dim}
    for d = 1:Dim
        F_d = let d=d
            @inline (x,y)->F(x,y)[d]
        end
        _hadamard_sum_ATr!(rhs,ATr_list[d],ScalingMatrices[d],F_d,u)
    end
end

@inline function _hadamard_sum_ATr!(rhs, ATr::AbstractSparseMatrix, ScalingMatrix, F, u)
    n = size(ATr,2)
    rows = rowvals(ATr)
    vals = nonzeros(ATr)
    for i = 1:n 
        ui = u[i]
        val_i = rhs[i] 
        for id in nzrange(ATr,i)
            j   = rows[id]
            Aij = vals[id]
            val_i += Aij * ScalingMatrix[j,i] * F(ui,u[j])
        end
        rhs[i] = val_i
    end
end


