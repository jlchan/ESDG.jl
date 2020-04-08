module SimpleBlockSparseMatrices

using LinearAlgebra
using SparseArrays
#import Base.show

export SparseMatrixSimpleBSR, SparseMatrixCSC!

# SimpleBSR: assumes uniform block sizes + same number of blocks per row
# some ideas taken from Kristoffer Carlsson's BlockSparseMatrices.jl package
struct SparseMatrixSimpleBSR{Tv,Ti <: Integer}
    blocksize::Int            # number of rows/columns in a block
    colindices::Array{Ti,2}   # column indices of blocks - e.g., A[i,rowval[i]] has a block.
    nzval::Array{Tv, 3}       # Nonzero values, one "matrix" per block, nzval[i, j, block]

    I::Array{Ti,3}            # precomputed row indices for faster sparse construction
    J::Array{Ti,3}            # precomputed col indices for faster sparse construction
    CSCpermuted_indices::Array{Ti,1} # list of indices
end


# initialize SimpleBSR matrix of zeros
function SparseMatrixSimpleBSR(blocksize::Integer, colindices::Array{Ti,2}) where {Ti}
    nblocks = prod(size(colindices))

    local_ids = (1:blocksize)
    global_ids(offset) = local_ids .+ (offset-1)*blocksize
    Block(e1,e2) = CartesianIndices((global_ids(e1),global_ids(e2))) # emulating BlockArrays

    # store indices IJK - faster than allocating block by block
    I = zeros(Int,blocksize,blocksize,nblocks)
    J = similar(I)
    r,c = size(colindices)
    for i = 1:r
        for j = 1:c
            block_id = j + (i-1)*c
            block_cartesian_ids = Block(i,colindices[i,j])
            I[:,:,block_id] .= (x->x[1]).(block_cartesian_ids)
            J[:,:,block_id] .= (x->x[2]).(block_cartesian_ids)
        end
    end

    # fast CSC conversion: convert to colptr
    # loop through all active columns
    flattened_indices = reshape(1:blocksize*blocksize*nblocks,blocksize,blocksize,nblocks) # UnitRange avoids allocation
    CSCpermuted_indices = zeros(Int,blocksize,blocksize*nblocks) # [col1, col2, ..., col_blocksize*nblocks]

    sk = 1
    for blockcol in unique(colindices[:])
        # find i,j indices of each block in a col
        blocks_in_column = findall(blockcol .== colindices)

        # sort by row ids
        p = sortperm((x->x[1]).(blocks_in_column))
        blocks_in_column .= blocks_in_column[p]

        # loop over each column
        for col = 1:blocksize
            for cartindex in blocks_in_column # sorts by row first
                block_id = cartindex[2] + (cartindex[1]-1)*size(colindices,2)
                CSCpermuted_indices[:,sk] .= flattened_indices[:,col,block_id]
                sk += 1
            end
        end
    end
    # CSCpermuted_indices = reshape(CSCpermuted_indices,blocksize,blocksize,nblocks)

    return SparseMatrixSimpleBSR(blocksize,colindices,zeros(blocksize,blocksize,nblocks),I,J,CSCpermuted_indices[:])
end

# convert from SimpleBSR to SparseMatrixCSC
# slower but usually faster than the actual solve
function SparseArrays.SparseMatrixCSC(A::SparseMatrixSimpleBSR{Tv, Ti}) where {Tv, Ti <: Integer}
    nblockrows = size(A.colindices,1)
    nblocks    = prod(size(A.colindices))
    n          = A.blocksize*nblockrows
    return sparse(A.I[:],A.J[:],A.nzval[:],n,n)
end

# in-place conversion from SimpleBSR to SparseMatrixCSC
function SparseMatrixCSC!(B::SparseMatrixCSC,A::SparseMatrixSimpleBSR{Tv, Ti}) where {Tv, Ti <: Integer}
    B.nzval .= A.nzval[:][A.CSCpermuted_indices] # need to add permutation, but this should be super fast
end

end # module
