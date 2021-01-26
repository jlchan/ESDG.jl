
function precompute(rd::RefElemData,md::MeshData)
    @unpack rxJ,sxJ,ryJ,syJ,nxJ,nyJ,J,sJ,mapP  = md
    @unpack Dr,Ds,M,LIFT,Vf = rd

    # precompute for Cartesian mesh
    Qr,Qs = M*Dr, M*Ds
    Qx = @. rxJ[1]*Qr + sxJ[1]*Qs
    Qy = @. ryJ[1]*Qr + syJ[1]*Qs
    SxTr,SyTr = map(A->droptol!(sparse(transpose(M\(A-A'))),1e-12),(Qx,Qy)) # flip sign for skew transpose

    # build Fmask to extract face ids
    Vf_ids = findall(@. abs(Vf) > 1e-12)
    Fmask = zeros(Int64,size(Vf,1))
    for ij in Vf_ids
        Fmask[ij[1]] = ij[2]
    end

    return SxTr,SyTr,Fmask,diag(M)
end

function compute_logs(Q)
    Qprim = cons_to_prim_beta(Euler{2}(),Q)  # WARNING: global
    return (Qprim[1],Qprim[2],Qprim[3],Qprim[4],log.(Qprim[1]),log.(Qprim[4])) # append logs
end

f2D(QL,QR) = fS_prim_log(Euler{2}(),QL,QR)

# compute surface flux + lifted contributions
function surface_contributions!(rhsQ,Qlog,rd,md,Fmask)
    @unpack LIFT = rd
    @unpack mapP,nxJ,nyJ = md
    Qflog = map(x->x[Fmask,:],Qlog) # WARNING: global
    QPlog = map(x->x[mapP],Qflog)   # WARNING: global
    flux = @SVector [zeros(size(nxJ,1)) for fld = 1:4] # tmp storage
    for e = 1:size(nxJ,2)
        for i = 1:size(nxJ,1)
            fi = f2D(getindex.(Qflog,i,e),getindex.(QPlog,i,e))
            fni = nxJ[i].*fi[1] + nyJ[i].*fi[2]
            setindex!.(flux,fni,i)
        end
        map((x,y)->x .+= LIFT*y,view.(rhsQ,:,e),flux) # WARNING: LIFT can be optimized
    end
    return nothing
end

function volume_contributions!(rhsQ,Qlog,SxTr,SyTr)
    for e = 1:size(first(Qlog),2)
        hadamard_sum_ATr!(view.(rhsQ,:,e),(SxTr,SyTr),f2D,view.(Qlog,:,e)) # overwrites
    end
    return nothing
end

function rhs!(rhsQ,Q,md,rd,SxTr,SyTr,Fmask)
    fill!.(rhsQ,zero(eltype(Q)))
    SxTr,SyTr,Fmask,wq = precompute(rd,md)
    Qlog = compute_logs(Q)
    surface_contributions!(rhsQ,Qlog,rd,md,Fmask)
    volume_contributions!(rhsQ,Qlog,SxTr,SyTr)
    (x-> x ./= -md.J[1,1]).(rhsQ)
    return nothing
end

function initial_condition(md)
    @unpack x,y = md
    rho = @. 2 + .0*exp(-25*(x^2+y^2))
    # rho = @. 2 + .1*sin(pi*x)*sin(pi*y)
    u = .5*ones(size(x))
    v = 0*ones(size(x))
    p = one.(rho) # rho.^Euler{2}().γ
    Q = VectorOfArray(SVector{4}(prim_to_cons(Euler{2}(),(rho,u,v,p))...))
    return Q
end
