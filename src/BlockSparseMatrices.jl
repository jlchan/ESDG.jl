
module BlockSparseMatrices

using LinearAlgebra
using SparseArrays
# import Base: Order.Forward

# include Block type for indexing
import BlockArrays.Block,BlockArrays.BlockIndex
export Block,BlockIndex # for block and global indexing

export SparseMatrixBSC, SparseMatrixCSC!, getCSCordering # for fast BSC-CSC conversion
export block_spzeros
export getBlockIndices # gets list of Block(i,j) indices for non-zero blocks
export block_lrmul # for BLAS-3 left/right multiply (B^T*A_i*B reductions)
# export blocksize, getCSCindex
export nnzblocks, nblocks, nzblockrange

include("SparseMatrixBSC.jl")

end # module
