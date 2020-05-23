"""
build_node_maps(Xf,FToF)

Intialize the connectivity table along all edges and boundary node tables of all
elements. mapM - map minus (interior). mapP - map plus (exterior).

Xf = (xf,yf,zf) and FToF is size (Nfaces*K) and FToF[face] = face neighbor

mapM,mapP are size Nfp x (Nfaces*K)

# Examples
```jldoctest
mapM,mapP,mapB = build_node_maps((xf,yf),FToF)
```
"""
function build_node_maps(xf,yf,FToF)
    build_node_maps((xf,yf),FToF)
end
function build_node_maps(xf,yf,zf,FToF)
    build_node_maps((xf,yf,zf),FToF)
end

function build_node_maps(Xf,FToF)

    NfacesK = length(FToF)
    NODETOL = 1e-10;
    dims = length(Xf)

    # number nodes consecutively
    Nfp  = convert(Int,length(Xf[1]) / NfacesK)
    mapM = reshape(collect(1:length(Xf[1])), Nfp, NfacesK);
    mapP = copy(mapM);

    ids = collect(1:Nfp)
    for (f1,f2) in enumerate(FToF)

        # find find volume node numbers of left and right nodes
        D = zeros(Nfp,Nfp)
        for i = 1:dims
            Xfi = reshape(Xf[i],Nfp,NfacesK)
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
