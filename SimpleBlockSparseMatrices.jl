module SimpleBlockSparseMatrices

using LinearAlgebra
using SparseArrays
#import Base.show

export SparseMatrixSimpleBSR

# SimpleBSR: assumes uniform block sizes + same number of blocks per row
# some ideas from Kristoffer Carlsson's BlockSparseMatrices.jl package
struct SparseMatrixSimpleBSR{Tv,Ti <: Integer}
    blocksize::Int            # number of rows/columns in a block
    colindices::Array{Ti,2}   # column indices of blocks - e.g., A[i,rowval[i]] has a block.
    nzval::Array{Tv, 3}       # Nonzero values, one "matrix" per block, nzval[i, j, block]
end



# initialize empty simple BSR matrix
function SparseMatrixSimpleBSR(blocksize::Integer, colindices::Array{Ti,2}) where {Ti}
    nblocks = prod(size(colindices))
    return SparseMatrixSimpleBSR(blocksize,colindices,zeros(blocksize,blocksize,nblocks))
end

# convert from SimpleBSR to SparseMatrixCSC
function SparseArrays.SparseMatrixCSC(A::SparseMatrixSimpleBSR{Tv, Ti}) where {Tv, Ti <: Integer}

    nblockrows = size(A.colindices,1)
    nblocks = prod(size(A.colindices))
    local_ids = (1:A.blocksize)
    global_ids(offset) = local_ids .+ (offset-1)*A.blocksize
    Block(e1,e2) = CartesianIndices((global_ids(e1),global_ids(e2))) # emulating BlockArrays

    B = spzeros(A.blocksize*nblocks,A.blocksize*nblocks)

    # determine i,j,val
    # I = zeros(Int,A.blocksize,A.blocksize,nblocks)
    # J = similar(I)
    r,c = size(A.colindices)
    for i = 1:r
        for j = 1:c
            block_id = j + (i-1)*c
            B[Block(i,A.colindices[i,j])] .= A.nzval[:,:,block_id]
            # block_cartesian_ids = Block(i,A.colindices[i,j])
            # I[:,:,block_id] .= (x->x[1]).(block_cartesian_ids)
            # J[:,:,block_id] .= (x->x[2]).(block_cartesian_ids)
        end
    end


    return B
    # return sparse(I[:],J[:],A.nzval[:],A.blocksize*nblockrows,A.blocksize*nblockrows)
end

end # module
