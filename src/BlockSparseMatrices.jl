
module BlockSparseMatrices

using LinearAlgebra
using SparseArrays
# import Base: Order.Forward

# include Block type for indexing
import BlockArrays.Block,BlockArrays.BlockIndex
export Block,BlockIndex # for block and global indexing

export SparseMatrixBSC # for fast BSC-CSC conversion
export block_spzeros
export getBlockIndices # gets list of Block(i,j) indices for non-zero blocks
export block_lrmul, sparseBSC_diag_rmult! # for BLAS-3 left/right multiply (B*A_i*C), scaling A::SparseBSC*D::Diagonal
# export blocksize, getCSCindex
export nnzblocks, nblocks, nzblockrange

include("SparseMatrixBSC.jl")

end # module
