"""
Module CommonUtils

General purpose utilities usable by all element types

"""

module ExplicitJacobians

using LinearAlgebra # for I matrix in geometricFactors
using ForwardDiff
using StaticArrays
using SparseArrays
using UnPack
using SetupDG
using BlockSparseMatrices

export init_jacobian_matrices
export hadamard_jacobian,accum_hadamard_jacobian!
export reduce_jacobian!
export hadamard_sum, hadamard_sum!
export banded_matrix_function
export columnize

# for constructing DG matrices
export build_rhs_matrix, assemble_global_SBP_matrices_2D

# use w/reshape to convert from matrices to arrays of arrays
columnize(A) = SVector{size(A,2)}([A[:,i] for i in 1:size(A,2)])

#################################################################
#####  block SparseBSC version for faster dense linear algebra
##################################################################


# block SparseBSC version of the jacobian assembly
function hadamard_jacobian(Q::SparseMatrixBSC, dF, U::AbstractArray, scale = -1)

    Nfields = length(U)
    #A = SMatrix{Nfields,Nfields}([block_spzeros(Q) for i = 1:Nfields, j=1:Nfields]) # why does StaticArrays break?
    A = [block_spzeros(Q) for i = 1:Nfields, j=1:Nfields] # why does StaticArrays break?
    accum_hadamard_jacobian!(A,Q,dF,U,scale)
    return A
end

# compute and accumulate contributions from a Jacobian function dF
#function accum_hadamard_jacobian!(A::SMatrix{Nfields,Nfields,SparseMatrixBSC{Tv,Ti}}, Q::SparseMatrixBSC{Tv,Ti},
                                  # dF, U::AbstractArray, scale = -1) where {Nfields,Tv,Ti <: Integer}
function accum_hadamard_jacobian!(A::Matrix{SparseMatrixBSC{Tv,Ti}}, Q::SparseMatrixBSC{Tv,Ti},
                                  dF, U::AbstractArray, scale = -1) where {Tv,Ti <: Integer}

    Nfields = length(U)

    # local storage
    R,C = BlockSparseMatrices.blocksize(Q)
    Alocal = SMatrix{Nfields,Nfields}([zeros(R,C) for i = 1:Nfields,j=1:Nfields])

    # loop over blocks
    for block_id in getBlockIndices(Q)

        Qblock = Q[block_id] # dense block extracted in order

        fill!.(Alocal,zero(Tv))
        for j = 1:size(Qblock,2) # access column major

            j_global = j + (block_id.n[2]-1)*Q.C # should fix - avoid using internals of Block...
            Uj = getindex.(U,j_global)

            for i = 1:size(Qblock,1)

                i_global = i + (block_id.n[1]-1)*Q.R
                Ui = getindex.(U,i_global)

                dFdU = dF(Ui,Uj) # local Jacobian blocks
                Qij  = Qblock[i,j]

                for n = 1:Nfields, m = 1:Nfields
                    Alocal[m,n][i,j] += dFdU[m,n]*Qij
                    #A[m,n][block_id][i,j] += dFdU[m,n]*Qij
                end
            end
        end

        # store local blocks back in A
        for n = 1:Nfields, m = 1:Nfields
            A[m,n][block_id] += Alocal[m,n]
        end
    end

    # add diagonal entry assuming Q = +/- Q^T
    for m = 1:Nfields, n = 1:Nfields
        Asum = scale * vec(sum(A[m,n],dims=1))

        # accumulate field block sum into diagonal blocks
        # assumes that there are diagonal blocks allocated in Q::SparseBSC (should be for DG)
        # assumes that A[m,n] has same sparsity structure as Q
        for i = 1:BlockSparseMatrices.nblocks(Q,2)
            A[m,n][Block(i,i)] += diagm(Asum[(1:C) .+ (i-1)*C])
        end
    end
end

# # computes block-banded matrix whose bands are entries of matrix-valued
# # function evals (e.g., a Jacobian function).
# function banded_matrix_function(mat_fun,U::AbstractArray)
#     Nfields = length(U)
#     num_pts = length(U[1])
#
#     A = spzeros(Nfields*num_pts,Nfields*num_pts)
#     ids(m) = (1:num_pts) .+ (m-1)*num_pts
#     Block(m,n) = CartesianIndices((ids(m),ids(n)))
#
#     for i = 1:num_pts
#         mat_i = mat_fun(getindex.(U,i))
#         for n = 1:Nfields, m = 1:Nfields
#             A[Block(m,n)[i,i]] = mat_i[m,n] # TODO: replace with fast sparse constructor
#         end
#     end
#     return A
# end



#################################################################
#####  regular SparseCSC version for faster dense linear algebra
##################################################################

# sparse matrix assembly version
# can only deal with one coordinate component at a time in higher dimensions
function hadamard_jacobian(Q::SparseMatrixCSC, dF, U::AbstractArray, scale = -1)

    Nfields = length(U)
    NpK = size(Q,2)
    blockIds = repeat([NpK],Nfields)
    A = spzeros(NpK*Nfields,NpK*Nfields)

    accum_hadamard_jacobian!(A,Q,dF,U,scale)

    return A
end

# compute and accumulate contributions from a Jacobian function dF
function accum_hadamard_jacobian!(A::SparseMatrixCSC, Q::SparseMatrixCSC,
    dF, U::AbstractArray, scale = -1)

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
            A[Block(m,n)[i,j]] += dFdU[m,n]*Qij
        end
    end

    # add diagonal entry assuming Q = +/- Q^T
    for m = 1:Nfields, n = 1:Nfields
        #Asum = scale * vec(sum(A[Block(m,n)],dims=1)) # CartesianIndices broken
        Asum = scale * vec(sum(A[ids(m),ids(n)],dims=1)) # TODO: fix slowness (maybe just alloc issue?)
        # A[Block(m,n)] .+= diagm(Asum) # can't index all at once - bug related to CartesianIndices?
        for i = 1:num_pts
            A[Block(m,n)[i,i]] += Asum[i]
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
            A[Block(m,n)[i,i]] = mat_i[m,n] # TODO: replace with fast sparse constructor
        end
    end
    return A
end

# init sparse matrices based on elem-to-elem connectivity
# dims = tuple of matrix dimensions for each matrix (assumes square matrices)
function init_jacobian_matrices(md::MeshData, dims, Nfields=1)
    @unpack FToF = md
    Nfaces,K = size(FToF)
    EToE = @. (FToF.-1) ÷ Nfaces + 1
    A = spzeros.(K.*dims,K.*dims)

    ids(e,block_size) = (@. (1:block_size) + (e-1)*block_size)

    for e = 1:K, f = 1:Nfaces
        enbr = EToE[f,e]

        for (i,block_size) = enumerate(dims)
            id,id_nbr = ids.((e,enbr), block_size)

            # init to non-zeros to retain sparsity pattern
            fill!(A[i][id, id], 1e-16)
            fill!(A[i][id, id_nbr], 1e-16)
        end
    end

    return repeat.(A,Nfields,Nfields)
end



# =============== for residual evaluation ================

# use ATr for faster col access of sparse CSC matrices
function hadamard_sum(ATr::SparseMatrixCSC,F,u::AbstractArray)
    m, n = size(ATr)
    # rhs = [zeros(n) for i in eachindex(u)]
    rhs = MVector{length(u)}([zeros(n) for i in eachindex(u)]) # probably faster w/StaticArrays?
    hadamard_sum!(rhs,ATr,F,u)
    return rhs
end

# computes ∑ A_ij * F(u_i,u_j) = (A∘F)*1 for flux differencing
function hadamard_sum!(rhs::AbstractArray,ATr::SparseMatrixCSC,F,u::AbstractArray)
    cols = rowvals(ATr)
    vals = nonzeros(ATr)
    m, n = size(ATr)
    for i = 1:n
        ui = getindex.(u,i)
        val_i = zeros(length(u))
        for j in nzrange(ATr, i) # column-major: extracts ith col of ATr = ith row of A
            col = cols[j]
            Aij = vals[j]
            uj = getindex.(u,col)
            val_i += Aij * F(ui,uj)
        end
        setindex!.(rhs,val_i,i)
    end
end


# ============== Constructing DG matrices =================

# flexible but slow function to construct global matrices based on rhs evals
# Np,K = number dofs and elements
# vargs = other args for applyRHS
function build_rhs_matrix(applyRHS,Np,K,vargs...)
    u = zeros(Np,K)
    A = spzeros(Np*K,Np*K)
    for i in eachindex(u)
        u[i] = 1
        r_i = applyRHS(u,vargs...)
        A[:,i] = droptol!(sparse(r_i[:]),1e-12)
        u[i] = 0
    end
    return A
end

# inputs = ref elem and mesh data
# Qrhskew,Qshskew = skew symmetric hybridized SBP operators
# note that md::MeshData needs FToF to also be periodic
function assemble_global_SBP_matrices_2D(rd::RefElemData, md::MeshData,
    Qrhskew, Qshskew, dtol=1e-12)

    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,mapP,FToF = md
    @unpack Nfaces,wf,wq = rd
    Nh = size(Qrhskew,1)
    NfqNfaces,K = size(nxJ)
    Nfq = convert(Int,NfqNfaces/Nfaces)

    EToE = @. (FToF.-1) ÷ Nfaces + 1
    mapPerm = ((mapP.-1) .% NfqNfaces) .+ 1 # mod out face offset
    fids = length(wq)+1:Nh # last indices correspond to face nodes

    ids(e) = (1:Nh) .+ (e-1)*Nh # block offsets
    Block(e1,e2) = CartesianIndices((ids(e1),ids(e2))) # emulating BlockArrays, but faster
    face_ids = (x,f)->reshape(x,Nfq,Nfaces)[:,f]

    Ax,Ay,Bx,By = ntuple(x->spzeros(Nh*K,Nh*K),4)
    for e = 1:K # loop over elements

        # self-contributions - assume affine for now
        Ax_local = rxJ[1,e]*Qrhskew + sxJ[1,e]*Qshskew
        Ay_local = ryJ[1,e]*Qrhskew + syJ[1,e]*Qshskew
        Ax[Block(e,e)] .= sparse(Ax_local)
        Ay[Block(e,e)] .= sparse(Ay_local)

        # neighboring contributions (ignore self-neighbors)
        for (f,enbr) in enumerate(EToE[:,e])
            if enbr!=e
                fperm = face_ids(reshape(mapPerm[:,e],Nfq,Nfaces),f)
                Axnbr = spdiagm(0 => face_ids(.5*wf.*nxJ[:,e],f))
                Aynbr = spdiagm(0 => face_ids(.5*wf.*nyJ[:,e],f))
                Bx[Block(e,enbr)[face_ids(fids,f),fids[fperm]]] .= Axnbr # TODO: switch to block access pattern
                By[Block(e,enbr)[face_ids(fids,f),fids[fperm]]] .= Aynbr
            end
        end
    end
    return droptol!.((Ax,Ay,Bx,By),dtol)
end

end
