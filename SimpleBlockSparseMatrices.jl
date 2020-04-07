module SimpleBlockSparseMatrices

using LinearAlgebra
using SparseArrays
#import Base.show

export SparseMatrixSimpleBSR

# SimpleBSR: assumes uniform block sizes + same number of blocks per row
struct SparseMatrixSimpleBSR{Tv,Ti <: Integer}
    blocksize::Int            # number of rows/columns in a block
    colindices::Array{Ti,2}   # column indices of blocks - e.g., A[i,rowval[i]] has a block.
    nzval::Array{Tv, 3}       # Nonzero values, one "matrix" per block, nzval[i, j, block]
end

# initialize empty simple BSR matrix
function SparseMatrixSimpleBSR(blocksize::Integer, colindices::Array{Ti,2}) where {Ti}
    return SparseMatrixSimpleBSR(blocksize,colindices,zeros(blocksize,blocksize,prod(size(colindices))))
end

# convert from SimpleBSR to SparseMatrixCSC
function SparseArrays.SparseMatrixCSC(A::SparseMatrixSimpleBSR{Tv, Ti}) where {Tv, Ti <: Integer}
    nblocks = size(A.colindices,1)
    B = spzeros(A.blocksize*nblocks,A.blocksize*nblocks)

    local_ids = (1:A.blocksize)
    global_ids(offset) = local_ids .+ (offset-1)*A.blocksize
    Block(e1,e2) = CartesianIndices((global_ids(e1),global_ids(e2))) # emulating BlockArrays

    r,c = size(A.colindices)
    for i = 1:r
        for j = 1:c
            block_id = j + (i-1)*c
            B[Block(i,A.colindices[i,j])] .= A.nzval[:,:,block_id]
        end
    end

    return B
end

end # module
