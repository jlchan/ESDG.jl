module AssembleGlobalSBPMatrices

using LinearAlgebra
using SparseArrays
using UnPack

using SetupDG

# for constructing global DG matrices
export build_rhs_matrix, assemble_global_SBP_matrices_2D

"""
function build_rhs_matrix(applyRHS,Np,K,vargs...)

flexible but slow function to construct global matrices based on rhs evals
applyRHS = function to evaluate rhs f(u(t)) given a solution vector u(t)
Np,K = number dofs and elements
vargs = other args for applyRHS

# Examples
```jldoctest
"""
function build_rhs_matrix(applyRHS,Np,K,vargs...)
    u = zeros(Np,K)
    A = spzeros(Np*K,Np*K)
    for i in eachindex(u)
        u[i] = one(eltype(u))
        r_i = applyRHS(u,vargs...)
        A[:,i] = droptol!(sparse(r_i[:]),1e-12)
        u[i] = zero(eltype(u))
    end
    return A
end

"""
function assemble_global_SBP_matrices_2D(rd::RefElemData, md::MeshData,
                                         Qrhskew, Qshskew, dtol=1e-12)

rd,md = ref elem and mesh data
Ax,Ay,Bx,By = global operators corresponding to vol, surface terms of DG derivatives.
B = global operator for surface term with only sJ

note that md::MeshData needs FToF to also be periodic

# Examples
```jldoctest
"""

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
