"""
function build_periodic_boundary_maps(xf,yf,LX,LY,mapM,mapP,mapB)
"""

# version which mods FToF as well
function build_periodic_boundary_maps!(xf,yf,LX,LY,NfacesTotal,mapM,mapP,mapB,FToF)

        # find boundary faces (e.g., when FToF[f] = f)
        Flist = 1:length(FToF)
        Bfaces = findall(vec(FToF) .== Flist)

        xb = xf[mapB]
        yb = yf[mapB]
        Nfp = convert(Int,length(xf)/NfacesTotal)
        Nbfaces = convert(Int,length(xb)/Nfp)
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
        NODETOL = 1e-12
        yfaces = map(x->x[1],findall(@. (@. abs(yc-ymax)<NODETOL*LY) | (@. abs(yc-ymin)<NODETOL*LY)))
        xfaces = map(x->x[1],findall(@. (@. abs(xc-xmax)<NODETOL*LX) | (@. abs(xc-xmin)<NODETOL*LX)))

        # find matches in y faces
        for i = yfaces
            for j = yfaces
                if i!=j
                    if abs(xc[i]-xc[j])<NODETOL*LX && abs(abs(yc[i]-yc[j])-LY)<NODETOL*LY
                        Xa,Xb = meshgrid(xb[:,i],xb[:,j])
                        D = @. abs(Xa-Xb)
                        ids = map(x->x[1],findall(@.D < NODETOL*LX))
                        mapPB[:,i]=mapMB[ids,j]

                        FToF[Bfaces[i]] = Bfaces[j]
                    end
                end
            end
        end

        # find matches in x faces
        for i = xfaces
            for j = xfaces
                if i!=j
                    if abs(yc[i]-yc[j])<NODETOL*LY && abs(abs(xc[i]-xc[j])-LX)<NODETOL*LX
                        Ya,Yb = meshgrid(yb[:,i],yb[:,j])
                        D = @. abs(Ya-Yb)
                        ids = map(x->x[1],findall(@.D < NODETOL*LY))
                        mapPB[:,i]=mapMB[ids,j]

                        FToF[Bfaces[i]] = Bfaces[j]
                    end
                end
            end
        end

        return mapPB[:]
end

function build_periodic_boundary_maps(xf,yf,LX,LY,NfacesTotal,mapM,mapP,mapB)

    FToF = collect(1:NfacesTotal) # dummy argument
    mapPB = build_periodic_boundary_maps!(xf,yf,LX,LY,NfacesTotal,mapM,mapP,mapB,FToF)
    return mapPB[:]

end

function build_periodic_boundary_maps(xf,yf,zf,LX,LY,LZ,NfacesTotal,mapM,mapP,mapB)

    xb = xf[mapB]
    yb = yf[mapB]
    zb = zf[mapB]
    Nfp = convert(Int,length(xf)/NfacesTotal)
    Nbfaces = convert(Int,length(xb)/Nfp)
    xb = reshape(xb,Nfp,Nbfaces)
    yb = reshape(yb,Nfp,Nbfaces)
    zb = reshape(zb,Nfp,Nbfaces)

    # compute centroids of faces
    xc = vec(sum(xb,dims=1)/Nfp)
    yc = vec(sum(yb,dims=1)/Nfp)
    zc = vec(sum(zb,dims=1)/Nfp)
    mapMB = reshape(mapM[mapB],Nfp,Nbfaces)
    mapPB = reshape(mapP[mapB],Nfp,Nbfaces)

    xmax = maximum(xc);  xmin = minimum(xc)
    ymax = maximum(yc);  ymin = minimum(yc)
    zmax = maximum(zc);  zmin = minimum(zc)

    "determine which faces lie on x and y boundaries"
    NODETOL = 1e-12
    xfaces = map(x->x[1],findall(@. (@. abs(xc-xmax)<NODETOL*LX) | (@. abs(xc-xmin)<NODETOL*LX)))
    yfaces = map(x->x[1],findall(@. (@. abs(yc-ymax)<NODETOL*LY) | (@. abs(yc-ymin)<NODETOL*LY)))
    zfaces = map(x->x[1],findall(@. (@. abs(zc-zmax)<NODETOL*LZ) | (@. abs(zc-zmin)<NODETOL*LZ)))

    # find matches in x faces
    for i = xfaces
        for j = xfaces
            if i!=j
                if abs(yc[i]-yc[j])<NODETOL*LY && abs(zc[i]-zc[j])<NODETOL*LZ && abs(abs(xc[i]-xc[j])-LX)<NODETOL*LX
                    Ya,Yb = meshgrid(yb[:,i],yb[:,j])
                    Za,Zb = meshgrid(zb[:,i],zb[:,j])
                    D = @. abs(Ya-Yb) + abs(Za-Zb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LY))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    # find matches in y faces
    for i = yfaces
        for j = yfaces
            if i!=j
                if abs(xc[i]-xc[j])<NODETOL*LX && abs(zc[i]-zc[j])<NODETOL*LZ && abs(abs(yc[i]-yc[j])-LY)<NODETOL*LY
                    Xa,Xb = meshgrid(xb[:,i],xb[:,j])
                    Za,Zb = meshgrid(zb[:,i],zb[:,j])
                    D = @. abs(Xa-Xb) + abs(Za-Zb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LX))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    # find matches in y faces
    for i = zfaces
        for j = zfaces
            if i!=j
                if abs(xc[i]-xc[j])<NODETOL*LX && abs(yc[i]-yc[j])<NODETOL*LY && abs(abs(zc[i]-zc[j])-LZ)<NODETOL*LZ
                    Xa,Xb = meshgrid(xb[:,i],xb[:,j])
                    Ya,Yb = meshgrid(yb[:,i],yb[:,j])
                    D = @. abs(Xa-Xb) + abs(Ya-Yb)
                    ids = map(x->x[1],findall(@.D < NODETOL*LX))
                    mapPB[:,i]=mapMB[ids,j]
                end
            end
        end
    end

    return mapPB[:]
end
