"""
Module RefinementTree

A *very* simplistic implementation for adaptive quad meshing
"""

module RefinementTree

export Cell, MeshTree
export refine!, enforce_one_irregularity!
export get_level

"Cell structure"
mutable struct Cell
    index
    parent
    children
    childId
end

"Tree of cells for an isotropic refined mesh"
mutable struct MeshTree
    active
    tree
    EToE
    EToF
    FToF

    "make coarse mesh quadtree"
    function MeshTree(EToE,EToF)
        K = size(EToE,1)
        tree = []
        for ee = 1:K
            push!(tree,Cell(ee,ee,nothing,nothing))
        end
        active = trues(K)
        return new(active,tree,EToE,EToF,nothing)
    end

    "make coarse mesh quadtree"
    function MeshTree(FToF)
        Nfaces,K = size(FToF,1)
        tree = []
        for ee = 1:K
            push!(tree,Cell(ee,ee,nothing,nothing))
        end
        active = trues(K)
        return new(active,tree,nothing,nothing,FToF)
    end
end

function refine!(e,meshtree)
    if meshtree.active[e]==false
        error("Cannot refine element: is inactive.")
    end
    num_children = 4 # assume quad
    Nfaces = 4
    K = length(meshtree.tree)
    newnbr = K .+ collect(1:num_children)
    meshtree.tree[e].children = newnbr
    for childId = 1:num_children
        push!(meshtree.tree,Cell(K+childId,e,nothing,childId))
    end

    EToE = meshtree.EToE
    EToF = meshtree.EToF
    EToE = vcat(EToE,zeros(Int,4,Nfaces))
    EToF = vcat(EToF,zeros(Int,4,Nfaces))

    # local neighbors
    EToE[K+1,2] = newnbr[2]; EToE[K+1,3] = newnbr[4]
    EToE[K+2,4] = newnbr[1]; EToE[K+2,3] = newnbr[3]
    EToE[K+3,1] = newnbr[2]; EToE[K+3,4] = newnbr[4]
    EToE[K+4,1] = newnbr[1]; EToE[K+4,2] = newnbr[3]
    #          nbr 3
    #        ---- ----
    #       | e4 | e3 |
    #  nbr4  ---- ----   nbr 2
    #       | e1 | e2 |
    #        ---- ----
    #          nbr 1
    EToF[K+1,2] = 4; EToF[K+1,3] = 1;
    EToF[K+2,4] = 2; EToF[K+2,3] = 1;
    EToF[K+3,1] = 3; EToF[K+3,4] = 2;
    EToF[K+4,1] = 3; EToF[K+4,2] = 4;

    ee = [[1 2],[2 3],[3 4],[4 1]] # new refined elements adjacent to each face
    en = [[4 3],[1 4],[2 1],[3 2]] # neighbors adjacent to each new elem
    fn = [3 4 1 2]; # faces of neighbors adjacent to each new elem
    for f = 1:4
        for i = 1:2 # neighbors per face
            new_elem = K + ee[f][i];

            if meshtree.tree[EToE[e,f]].children==nothing
                # if neighbor is not refined, connect to parent
                parent = meshtree.tree[EToE[e,f]].parent
                EToE[new_elem,f] = parent
                EToF[new_elem,f] = EToF[parent,f]
            else # if neighbor refined, pick up neighbors from children

                nbr_elem = meshtree.tree[EToE[e,f]].children[en[f][i]]
                EToE[new_elem,f] = nbr_elem
                EToE[nbr_elem,fn[f]] = new_elem

                EToF[new_elem,f] = EToF[e,f]; # inherit face connectivity
            end
        end
    end

    meshtree.EToE = EToE
    meshtree.EToF = EToF

    append!(meshtree.active,trues(num_children))
    meshtree.active[e] = false
end

"get level of refinement"
function get_level(e,meshtree)
    next_elem = e;
    level = 0;
    at_top = false;
    while !at_top
        parent = meshtree.tree[next_elem].parent;
        if parent == next_elem
            at_top = true;
        else
            next_elem = parent;
            level = level + 1;
        end
    end
    return level
end


"check if mesh is 1-irregular"
function enforce_one_irregularity!(meshtree)
    Nfaces = 4
    not_one_irregular = true
    while not_one_irregular
        not_one_irregular = false
        elems_to_refine = []
        for e = findall(meshtree.active)
            for f = 1:Nfaces
                enbr = meshtree.EToE[e,f]
                if meshtree.tree[enbr].children != nothing # if refined
                    not_one_irregular=true
                end
            end
        end
    end
end


end
