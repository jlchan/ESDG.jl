"""
function build_periodic_boundary_maps_2D(xf,yf,LX,LY,mapM,mapP,mapB)

# Examples
```jldoctest
"""

function build_periodic_boundary_maps_2D(xf,yf,LX,LY,Nfaces,mapM,mapP,mapB)

    xb = xf[mapB]
    yb = yf[mapB]
    Nfp = convert(Int,size(xf,1)/Nfaces)
    Nbfaces = convert(Int,length(xb[:])/Nfp)
    xb = reshape(xb,Nfp,Nbfaces)
    yb = reshape(yb,Nfp,Nbfaces)

    # compute centroids of faces
    xc = vec(sum(xb,dims=1)/Nfp)
    yc = vec(sum(yb,dims=1)/Nfp)
    mapMB = reshape(mapM[mapB],Nfp,Nbfaces)
    mapPB = reshape(mapP[mapB],Nfp,Nbfaces)

    xmax = maximum(xc)
    xmin = minimum(xc)
    ymax = maximum(yc)
    ymin = minimum(yc)

    "determine which faces lie on x and y boundaries"
    NODETOL = 1e-10
    xfaces = map(x->x[1],findall(@. (@. abs(yc-ymax)<NODETOL*LY) | (@. abs(yc-ymin)<NODETOL*LY)))
    yfaces = map(x->x[1],findall(@. (@. abs(xc-xmax)<NODETOL*LX) | (@. abs(xc-xmin)<NODETOL*LX)))

    # find matches in x faces
    for i = xfaces
        for j = xfaces
            if i!=j
                if abs(xc[i]-xc[j])<NODETOL*LX && abs(abs(yc[i]-yc[j])-LY)<NODETOL*LY
                    Xa,Xb = meshgrid(xb[:,i],xb[:,j])
                    D = @. abs(Xa-Xb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LX))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    # find matches in y faces
    for i = yfaces
        for j = yfaces
            if i!=j
                if abs(yc[i]-yc[j])<NODETOL*LY && abs(abs(xc[i]-xc[j])-LX)<NODETOL*LX
                    Ya,Yb = meshgrid(yb[:,i],yb[:,j])
                    D = @. abs(Ya-Yb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LY))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    return mapPB
end
