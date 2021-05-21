"""
    function hadamard_sum_ATr!(rhs, ATr, F, u, skip_index=(i,j)->false)

Computes and accumulates `sum(A.*F(u,u'),dims=2)` into `rhs`. 
""" 
@inline function hadamard_sum_ATr!(rhs, ATr, F, u, skip_index=(i,j)->false)
    rows,cols = axes(ATr)
    for i in cols
        ui = u[i]
        val_i = rhs[i]
        for j in rows
            if !skip_index(i,j)
                val_i += ATr[j,i] * F(ui,u[j]) # breaks for tuples, OK for StaticArrays
            end
        end
        rhs[i] = val_i # why not .= here?
    end
end

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

# accumulate A.*F into rhs but take advantage of sparsity of A
@inline function hadamard_sum_ATr!(rhs, ATr::AbstractSparseMatrix, F, u)
    rows = rowvals(ATr)
    # vals = nonzeros(ATr)
    for i = 1:size(ATr,2) # all ops should be same length
        ui = u[i]
        val_i = rhs[i] # accumulate into existing rhs
        for row_id in nzrange(ATr,i)
            j = rows[row_id]
            val_i += A[i,j]*F(ui,u[j]) # can replace A[i,j] with vals[row_id] but not for lazy sparse combo
        end
        rhs[i] = val_i
    end
end


# struct SameSparsityMatrices{N,Tv,Ti}
#     matrices::NTuple{N,SparseMatrixCSC{Tv,Ti}}
# end

# function SameSparsityMatrices(matrices...)
#     ijvals = findnz.(matrices)
#     getindex.(ijvals,1)
# end

# # computes A[i,j] = ∑_k f(i,j)[k] * matrices[k][i,j]
# struct LazySparseLinearCombination{N,Tv,Ti,F <: Function} <: AbstractSparseMatrix{Tv,Ti}
#     matrices::NTuple{N,SparseMatrixCSC{Tv,Ti}}
#     f::F # function of (i,j), returns tuple of scalings
# end
# function LazySparseLinearCombination(matrices,scalings)
#     # todo: standardize and make sure all matrices have the same
#     return LazySparseLinearCombination(matrices,(i,j)->scalings)
# end
# # set_f(A::LazySparseLinearCombination)
# Base.size(A::LazySparseLinearCombination) = size(first(A.matrices))
# Base.getindex(A::LazySparseLinearCombination,i,j)  = sum(getindex.(A.matrices,i,j) .* A.f(i,j))
# SparseArrays.rowvals(A::LazySparseLinearCombination) = rowvals(first(A.matrices))



