"""
Module CommonUtils

General purpose utilities usable by all element types

"""

module ExplicitJacobians

using LinearAlgebra # for I matrix in geometricFactors
using ForwardDiff
using StaticArrays
using SparseArrays

export hadamard_jacobian,hadamard_jacobian!
export hadamard_sum, hadamard_sum!
export banded_matrix_function
export columnize

# use w/reshape to convert from matrices to arrays of arrays
columnize(A) = SVector{size(A,2)}([A[:,i] for i in 1:size(A,2)])

# sparse matrix assembly version
# can only deal with one coordinate component at a time in higher dimensions
function hadamard_jacobian(Q::SparseMatrixCSC,F,U::AbstractArray,scale = -1)

    Nfields = length(U)
    NpK = size(Q,2)
    blockIds = repeat([NpK],Nfields)
    A = spzeros(NpK*Nfields,NpK*Nfields)

    dF(uL,uR) = ForwardDiff.jacobian(uR->F(uL,uR),uR)

    hadamard_jacobian!(A,Q,dF,U,scale)

    return A
end

function hadamard_jacobian!(A::SparseMatrixCSC,Q::SparseMatrixCSC,
                            dF,U::AbstractArray,scale = -1)

    fill!(A,0.0) # should zero sparse entries but keep sparsity pattern

    Nfields = length(U)

    num_pts = size(Q,1)
    ids(m) = (1:num_pts) .+ (m-1)*num_pts
    Block(m,n) = CartesianIndices((ids(m),ids(n)))

    # loop over non-zero indices in Q
    Qnz = zip(findnz(Q)...)
    for (i,j,Qij) in Qnz
        Ui = getindex.(U,i)
        Uj = getindex.(U,j)
        dFdU = dF(Ui,Uj)
        for n = 1:length(U), m=1:length(U)
            A[Block(m,n)[i,j]] = dFdU[m,n]*Qij
        end
    end

    # add diagonal entry assuming Q = +/- Q^T
    for m = 1:Nfields, n = 1:Nfields
        #Asum = scale * vec(sum(A[Block(m,n)],dims=1))
        Asum = scale * vec(sum(A[ids(m),ids(n)],dims=1)) # hack since CartesianIndices seems broken?
        for i = 1:num_pts
            A[Block(m,n)[i,i]] += Asum[i] # can't index all at once - bug?
        end
    end
end

# computes block-banded matrix whose bands are entries of matrix-valued
# function evals (e.g., a Jacobian function).
function banded_matrix_function(mat_fun,U::AbstractArray)
    Nfields = length(U)
    num_pts = length(U[1])

    A = spzeros(Nfields*num_pts,Nfields*num_pts)
    ids(m) = (1:num_pts) .+ (m-1)*num_pts
    Block(m,n) = CartesianIndices((ids(m),ids(n)))

    for i = 1:num_pts
        mat_i = mat_fun(getindex.(U,i))
        for n = 1:Nfields, m = 1:Nfields
            A[Block(m,n)[i,i]] = mat_i[m,n]
        end
    end
    return A
end

# use ATr for faster col access of sparse CSC matrices
function hadamard_sum(ATr::SparseMatrixCSC,F,u::AbstractArray)
    m, n = size(ATr)
    # rhs = [zeros(n) for i in eachindex(u)]
    rhs = MVector{length(u)}([zeros(n) for i in eachindex(u)]) # probably faster w/StaticArrays?
    hadamard_sum!(ATr,F,u,rhs)
    return rhs
end

# computes ∑ A_ij * F(u_i,u_j) = (A∘F)*1 for flux differencing
function hadamard_sum!(ATr::SparseMatrixCSC,F,u::AbstractArray,rhs::AbstractArray)
    cols = rowvals(ATr)
    vals = nonzeros(ATr)
    m, n = size(ATr)
    for i = 1:n
        ui = getindex.(u,i)
        val_i = zeros(length(u))
        for j in nzrange(ATr, i)
            col = cols[j]
            Aij = vals[j]
            uj = getindex.(u,col)
            val_i += Aij * F(ui,uj)
        end
        setindex!.(rhs,val_i,i)
    end
end


end
