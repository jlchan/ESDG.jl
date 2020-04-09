#

module BlockSparseMatrices

using LinearAlgebra
using SparseArrays
import Base: Order.Forward

import BlockArrays.Block # for block indexing
export Block

export SparseMatrixBSC, SparseMatrixCSC!
export blocksize, getCSCindex
export nnzblocks, nblocks, nzblockrange
# export getblock, getblock!, setblock! # emulating BlockArrays

include("SparseMatrixBSC.jl")

end # module
