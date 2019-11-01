"""
 build_node_maps_2D(xf,yf,Nfaces,EToE,EToF)

Intialize the connectivity table along all edges and boundary node tables of all
elements. mapM - map minus (interior). mapP - map plus (exterior).

# Examples
```jldoctest
"""

# function build_node_maps(xf,yf,EToE,EToF)
#
#     K = size(EToE,1);
#     Nfaces = size(EToF,2)
#     NODETOL = 1e-10;
#
#     # number nodes consecutively
#     Nfp  = convert(Int,size(xf,1) / Nfaces)
#     mapM = reshape(collect(1:K*Nfp*Nfaces), Nfp, Nfaces, K);
#     mapP = copy(mapM);
#
#     onevec = ones(1, Nfp);
#
#     for k1=1:K
#         for f1=1:Nfaces
#             # find neighbor
#             k2 = EToE[k1,f1]
#             f2 = EToF[k1,f1]
#
#             if (k1==k2)
#                 # do nothing if its a boundary face
#                 mapP[:,f1,k1] = mapM[:,f1,k1];
#             else
#                 # println("k1 = ",k1, "f1 = ",f1)
#                 ids = collect(1:Nfp)
#                 idM = @. ids + (f1-1)*Nfp;
#                 idP = @. ids + (f2-1)*Nfp;
#
#                 # find find volume node numbers of left and right nodes
#                 x1 = xf[idM,k1]; y1 = yf[idM,k1];
#                 x2 = xf[idP,k2]; y2 = yf[idP,k2];
#                 x1 = x1*onevec;  y1 = y1*onevec
#                 x2 = x2*onevec;  y2 = y2*onevec
#
#                 # Compute distance matrix
#                 D = abs.(x1 - transpose(x2)) .+ abs.(y1-transpose(y2))
#                 refd = maximum(D[:])
#                 idM = map(x->x[1], findall(@. D < NODETOL*refd))
#                 idP = map(x->x[2], findall(@. D < NODETOL*refd))
#                 mapP[idM,f1,k1] = @. idP + (f2-1)*Nfp + (k2-1)*Nfaces*Nfp
#             end
#         end
#     end
#
#     mapB = map(x->x[1],findall(@. mapM[:]==mapP[:]))
#
#     return mapM[:],mapP[:],mapB[:]
# end

function build_node_maps(xf,yf,FToF)
    build_node_maps((xf,yf),FToF)
end
function build_node_maps(xf,yf,zf,FToF)
    build_node_maps((xf,yf,zf),FToF)
end

"build node maps using only Xf = (xf,yf,zf) and FToF (Nfaces x K) mapping"
function build_node_maps(Xf,FToF)

    Nfaces, K = size(FToF)
    NODETOL = 1e-10;
    dims = length(Xf)

    # number nodes consecutively
    Nfp  = convert(Int,size(Xf[1],1) / Nfaces)
    mapM = reshape(collect(1:K*Nfp*Nfaces), Nfp, Nfaces*K);
    mapP = copy(mapM);

    ids = collect(1:Nfp)
    for (f1,f2) in enumerate(FToF)

        # find find volume node numbers of left and right nodes
        D = zeros(Nfp,Nfp)
        for i = 1:dims
            Xfi = reshape(Xf[i],Nfp,Nfaces*K)
            X1i = repeat(Xfi[ids,f1],1,Nfp)
            X2i = repeat(Xfi[ids,f2],1,Nfp)
            # Compute distance matrix
            D += abs.(X1i - transpose(X2i))
        end

        refd = maximum(D[:])
        idM = map(id->id[1], findall(@. D < NODETOL*refd))
        idP = map(id->id[2], findall(@. D < NODETOL*refd))
        mapP[idM,f1] = @. idP + (f2-1)*Nfp
    end

    mapB = map(x->x[1],findall(@. mapM[:]==mapP[:]))
    return mapM,mapP,mapB
end
