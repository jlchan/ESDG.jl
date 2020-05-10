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
# using BlockSparseMatrices

export init_jacobian_matrices
export hadamard_jacobian,accum_hadamard_jacobian!,hadamard_scale!
export reduce_jacobian!
export hadamard_sum, hadamard_sum!
export banded_matrix_function
export columnize

# for constructing DG matrices
export build_rhs_matrix, assemble_global_SBP_matrices_2D

# use w/reshape to convert from matrices to arrays of arrays
columnize(A) = SVector{size(A,2)}([A[:,i] for i in 1:size(A,2)])

##  hadamard function matrix utilities

# can only deal with one coordinate component at a time in higher dimensions
# assumes that Q/F is a skew-symmetric/symmetric pair
function hadamard_jacobian(Q::SparseMatrixCSC, dF::Function,
                           U::AbstractArray, Fargs::AbstractArray ...; scale = -1)

    Nfields = length(U)
    NpK = size(Q,2)
    blockIds = repeat([NpK],Nfields)
    A = spzeros(NpK*Nfields,NpK*Nfields)

    accum_hadamard_jacobian!(A,Q,dF,U,Fargs...; scale=scale)

    return A
end

" computes the matrix A_ij = Q_ij * F(u_i,u_j)
if you add extra args, they are passed to F(ux,uy) via F(u_i,u_j,args_i,args_j)
"
function hadamard_scale!(A::SparseMatrixCSC, Q::SparseMatrixCSC, F::Function,
                        U::AbstractArray, Fargs::AbstractArray ...)

    Nfields = length(U)
    num_pts = size(Q,1)
    ids(m) = (1:num_pts) .+ (m-1)*num_pts
    Block(m,n) = CartesianIndices((ids(m),ids(n)))

    # loop over non-zero indices in Q
    Qnz = zip(findnz(Q)...)
    for (i,j,Qij) in Qnz
        Ui = getindex.(U,i)
        Uj = getindex.(U,j)

        Fij = F(Ui,Uj,getindex.(Fargs,i)...,getindex.(Fargs,j)...)
        for n = 1:length(U), m=1:length(U)
            A[Block(m,n)[i,j]] += Fij[m,n]*Qij
        end
    end
end

# compute and accumulate contributions from a Jacobian function dF
function accum_hadamard_jacobian!(A::SparseMatrixCSC, Q::SparseMatrixCSC,
                                  dF::Function, U::AbstractArray, Fargs::AbstractArray ...; scale = -1)

    # scale A_ij = Q_ij * F(ui,uj)
    hadamard_scale!(A,Q,dF,U,Fargs...)

    # add diagonal entry assuming Q = - Q^T
    Nfields = length(U)
    num_pts = size(Q,1)
    ids(m) = (1:num_pts) .+ (m-1)*num_pts
    for m = 1:Nfields, n = 1:Nfields
        Asum = sum(A[ids(m),ids(n)],dims=1)
        A[ids(m),ids(n)] += spdiagm(0=>scale * vec(Asum))
    end
end

# computes block-banded matrix whose bands are entries of matrix-valued
# function evals (e.g., a Jacobian function).
function banded_matrix_function(mat_fun::Function, U::AbstractArray, Fargs::AbstractArray ...)
    Nfields = length(U)
    num_pts = length(U[1])

    A = spzeros(Nfields*num_pts,Nfields*num_pts)
    ids(m) = (1:num_pts) .+ (m-1)*num_pts
    Block(m,n) = CartesianIndices((ids(m),ids(n)))

    for i = 1:num_pts
        mat_i = mat_fun(getindex.(U,i),getindex.(Fargs,i)...)
        for n = 1:Nfields, m = 1:Nfields
            A[Block(m,n)[i,i]] = mat_i[m,n] # TODO: replace with fast sparse constructor
        end
    end
    return A
end

# =============== for residual evaluation ================

# use ATr for faster col access of sparse CSC matrices
function hadamard_sum(ATr,F::Function,u::AbstractArray,Fargs::AbstractArray ...)
    m, n = size(ATr)
    # rhs = [zeros(n) for i in eachindex(u)]
    rhs = MVector{length(u)}([zeros(n) for i in eachindex(u)]) # probably faster w/StaticArrays?
    hadamard_sum!(rhs,ATr,F,u,Fargs...)
    return rhs
end

# computes ∑ A_ij * F(u_i,u_j) = (A∘F)*1 for flux differencing
# separate code from hadamard_scale!, since it's non-allocating
function hadamard_sum!(rhs::AbstractArray,ATr::SparseMatrixCSC,F::Function,
                        u::AbstractArray,Fargs::AbstractArray ...)
    cols = rowvals(ATr)
    vals = nonzeros(ATr)
    m, n = size(ATr)
    for i = 1:n
        ui = getindex.(u,i)
        val_i = zeros(length(u))
        Farg_i = getindex.(Fargs,i)
        for j in nzrange(ATr, i) # column-major: extracts ith col of ATr = ith row of A
            col = cols[j]
            Aij = vals[j]
            uj = getindex.(u,col)
            val_i += Aij * F(ui,uj,Farg_i...,getindex.(Fargs,col)...)
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
# Ax,Ay,Bx,By,B = global operators. B = only sJ
# note that md::MeshData needs FToF to also be periodic
function assemble_global_SBP_matrices_2D(rd::RefElemData, md::MeshData,
                                         Qrhskew, Qshskew, dtol=1e-12)

    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,sJ,mapP,FToF = md
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

    Ax,Ay,Bx,By,B = ntuple(x->spzeros(Nh*K,Nh*K),5)
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
                Bx[Block(e,enbr)[face_ids(fids,f),fids[fperm]]] .= Axnbr
                By[Block(e,enbr)[face_ids(fids,f),fids[fperm]]] .= Aynbr
                B[ Block(e,enbr)[face_ids(fids,f),fids[fperm]]] .= spdiagm(0 => face_ids(.5*wf.*sJ[:,e],f))
            end
        end
    end
    return droptol!.((Ax,Ay,Bx,By,B),dtol)
end

end
