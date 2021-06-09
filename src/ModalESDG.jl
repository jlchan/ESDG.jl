"""
    struct ModalESDG{N,DIM,ElemType,F1,F2,F3} 
        volume_flux::F1 
        interface_flux::F2
        interface_dissipation::F3
        cons2entropy::F4
        entropy2cons::F5
    end

    `volume_flux`,`interface_flux`,`interface_dissipation` expect arguments `flux(orientation)(u_ll,u_rr)`. 
    A convenience constructor using Trixi's flux functions is provided. 
"""
struct ModalESDG{DIM,ElemType,Tv,F1,F2,F3,F4,F5} 
    rd::RefElemData{DIM,ElemType,Tv}
    volume_flux::F1 
    interface_flux::F2
    interface_dissipation::F3
    cons2entropy::F4
    entropy2cons::F5
end

"""
    function ModalESDG(rd::RefElemData,
        trixi_volume_flux::F1,
        trixi_interface_flux::F2,
        trixi_interface_dissipation::F3,
        cons2entropy::F4,
        entropy2cons::F5,        
        equations) where {F1,F2,F3}
    
Initialize a ModalESDG solver with Trixi fluxes as arguments, where trixi_*_flux has the form of
    trixi_*_flux(u_ll,u_rr,orientation,equations)

"""
function ModalESDG(rd::RefElemData,
                   trixi_volume_flux::F1,
                   trixi_interface_flux::F2,
                   trixi_interface_dissipation::F3,
                   cons2entropy::F4,
                   entropy2cons::F5,
                   equations) where {F1,F2,F3,F4,F5}

    volume_flux, interface_flux, interface_dissipation = let equations=equations
        volume_flux(orientation) = (u_ll,u_rr)->trixi_volume_flux(u_ll,u_rr,orientation,equations)
        interface_flux(orientation) = (u_ll,u_rr)->trixi_interface_flux(u_ll,u_rr,orientation,equations)
        interface_dissipation(orientation) = (u_ll,u_rr)->trixi_interface_dissipation(u_ll,u_rr,orientation,equations)
        volume_flux,interface_flux,interface_dissipation
    end
    return ModalESDG(rd,volume_flux,interface_flux,interface_dissipation,cons2entropy,entropy2cons)
end

function Base.show(io::IO, solver::ModalESDG{DIM}) where {DIM}
    println("Modal ESDG solver in $DIM dimension with ")
    println("   volume flux           = $(solver.volume_flux.trixi_volume_flux)")
    println("   interface flux        = $(solver.interface_flux.trixi_interface_flux)")    
    println("   interface dissipation = $(solver.interface_dissipation.trixi_interface_dissipation)")        
    println("   cons2entropy          = $(solver.cons2entropy)")            
    println("   entropy2cons          = $(solver.entropy2cons)")                
end

function compute_entropy_projection!(Q,solver::ModalESDG,cache,eqn)
    @unpack rd = solver    
    @unpack Vq = rd
    @unpack VhP,Ph = cache
    @unpack Uq, VUq, VUh, Uh = cache

    # if this freezes, try resetThreads()
    StructArrays.foreachfield((uout,u)->matmul!(uout,Vq,u),Uq,Q)
    tmap!(u->solver.cons2entropy(u,eqn),VUq,Uq) # 77.5μs
    StructArrays.foreachfield((uout,u)->matmul!(uout,VhP,u),VUh,VUq)
    tmap!(v->solver.entropy2cons(v,eqn),Uh,VUh) # 327.204 μs

    Nh,Nq = size(VhP)
    Uf = view(Uh,Nq+1:Nh,:) # 24.3 μs

    return Uh,Uf
end

## ======================== 1D rhs codes ==================================

function create_cache(md::MeshData{1}, equations, solver::ModalESDG)

    # make skew symmetric versions of the operators"
    Qrh,VhP,Ph = hybridized_SBP_operators(rd)
    Qrhskew = .5*(Qrh-transpose(Qrh))
    QrhskewTr = typeof(Qrh)(Qrhskew')

    project_and_store! = let Ph=Ph
        (y,x)->mul!(y,Ph,x) # can't use matmul! b/c its applied to a subarray
    end

    # tmp variables for entropy projection
    nvars = nvariables(equations)
    Uq = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars))
    VUq = similar(Uq)
    VUh = StructArray{SVector{nvars,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvars))
    Uh = similar(VUh)

    rhse_threads = [similar(Uh[:,1]) for _ in 1:Threads.nthreads()]

    cache = (;QrhskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh,
            rhse_threads,            
            project_and_store!)

    return cache
end

# ======================= 2D rhs codes =============================

function create_cache(md::MeshData{2}, equations, solver::ModalESDG)

    # for flux differencing on general elements
    Qrh,Qsh,VhP,Ph = hybridized_SBP_operators(rd)
    # make skew symmetric versions of the operators"
    Qrhskew = .5*(Qrh-transpose(Qrh))
    Qshskew = .5*(Qsh-transpose(Qsh))
    QrhskewTr = Matrix(Qrhskew') # punt to dense for now - need rotation?
    QshskewTr = Matrix(Qshskew') 

    project_and_store! = let Ph=Ph
        (y,x)->mul!(y,Ph,x)
    end    

    # tmp variables for entropy projection
    nvars = nvariables(equations)
    Uq = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars))
    VUq = similar(Uq)
    VUh = StructArray{SVector{nvars,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvars))
    Uh = similar(VUh)

    is_boundary_node = zeros(Int,size(md.xf))
    is_boundary_node[md.mapB] .= 1 # change to boundary tag later

    # tmp cache for threading
    rhse_threads = [similar(Uh[:,1]) for _ in 1:Threads.nthreads()]

    cache = (;md,
            QrhskewTr,QshskewTr,VhP,Ph,
            Uq,VUq,VUh,Uh,
            rhse_threads,
            is_boundary_node,
            project_and_store!)

    return cache
end

# function rhs!(dQ, Q::StructArray, t, md::MeshData{1}, equations,
#                     initial_condition, boundary_conditions, source_terms,
#                     solver::ModalESDG, cache)

#     @unpack QrhskewTr,VhP,Ph = cache
#     @unpack project_and_store! = cache
#     @unpack rxJ,J,nxJ,sJ,mapP = md
#     rd = solver.rd
#     @unpack wf = rd # careful - only wf, wq are inferrable from rd.
#     @unpack volume_flux, interface_flux, interface_dissipation = solver
#     @unpack is_boundary_node = cache
    
#     Nh,Nq = size(VhP)
#     skip_index = let Nq=Nq
#         (i,j) -> i>Nq && j > Nq
#     end

#     Uh,Uf = compute_entropy_projection!(Q,solver,cache,equations) # N=2, K=16: 670 μs
        
#     @batch for e = 1:md.K 
#         rhse = cache.rhse_threads[Threads.threadid()]

#         fill!(rhse,zero(eltype(rhse))) # 40ns, (1 allocation: 48 bytes)
#         Ue = view(Uh,:,e)              # 8ns (0 allocations: 0 bytes) after @inline in Base.view(StructArray)
#         QxTr = LazyArray(@~ @. 2 * rxJ[1,e]*QrhskewTr )

#         hadamard_sum_ATr!(rhse, QxTr, volume_flux(1), Ue, skip_index) 
        
#         # add in interface flux contributions
#         for (i,vol_id) = enumerate(Nq+1:Nh)
#             UM, UP = Uf[i,e], Uf[mapP[i,e]]
#             Fx = interface_flux(1)(UP,UM)
#             diss = interface_dissipation(SVector{1}(nxJ[i,e]))(UM,UP)
#             val = (Fx * nxJ[i,e] + diss*sJ[i,e]) 
#             rhse[vol_id] = rhse[vol_id] + val
#         end        

#         # project down and store
#         @. rhse = -rhse/J[1,e]
        
#         StructArrays.foreachfield(project_and_store!,view(dQ,:,e),rhse) 
#     end

#     return nothing
# end

# function rhs!(dQ, Q::StructArray, t, md::MeshData{2}, equations,
#               initial_condition, boundary_conditions, source_terms,
#               solver::ModalESDG, cache)

#     @unpack project_and_store! = cache
#     @unpack QrhskewTr,QshskewTr,VhP = cache
#     @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
#     @unpack volume_flux, interface_flux, interface_dissipation = solver
#     @unpack wf = solver.rd

#     Nh,Nq = size(VhP)
#     skip_index = let Nq=Nq
#         (i,j) -> i > Nq && j > Nq
#     end

#     # @timeit_debug timer() "compute_entropy_projection!" begin
#     Uh,Uf = compute_entropy_projection!(Q,solver,cache,equations) # N=2, K=16: 670 μs
#     # end

#     @batch for e = 1:md.K
#         rhse = cache.rhse_threads[Threads.threadid()]

#         fill!(rhse,zero(eltype(rhse)))
#         Ue = view(Uh,:,e)   
#         QxTr = LazyArray(@~ @. 2 *(rxJ[1,e]*QrhskewTr + sxJ[1,e]*QshskewTr))
#         QyTr = LazyArray(@~ @. 2 *(ryJ[1,e]*QrhskewTr + syJ[1,e]*QshskewTr))

#         hadamard_sum_ATr!(rhse, QxTr, volume_flux(1), Ue, skip_index) 
#         hadamard_sum_ATr!(rhse, QyTr, volume_flux(2), Ue, skip_index) 

#         for (i,vol_id) = enumerate(Nq+1:Nh)
#             if cache.is_boundary_node[i,e] == 1
#             elseif cache.is_boundary_node[i,e]
#                 UM = Uf[i,e]
#                 ϱ, ϱu, ϱv, E = UM
#                 u, v = ϱu/ϱ, ϱv/ϱ
#                 nx, ny = nxJ[i,e]/sJ[i,e], nyJ[i,e]/sJ[i,e]
#                 u_n = u*nx + v*ny
#                 uP = u - 2*u_n*nx
#                 vP = v - 2*u_n*ny
#                 UP = SVector{4}(ϱ,ϱ*uP,ϱ*vP,E) 

#                 Fx = interface_flux(1)(UP,UM)
#                 Fy = interface_flux(2)(UP,UM)
#                 normal = SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e]
#                 diss = interface_dissipation(normal)(UM,UP)
#                 val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
#             else
#                 UM, UP = Uf[i,e], Uf[mapP[i,e]]
#                 Fx = interface_flux(1)(UP,UM)
#                 Fy = interface_flux(2)(UP,UM)
#                 normal = SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e]
#                 diss = interface_dissipation(normal)(UM,UP)
#                 val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
#             end            
#             rhse[vol_id] += val
#         end

#         @. rhse = -rhse / J[1,e]

#         # project down and store
#         StructArrays.foreachfield(project_and_store!, view(dQ,:,e), rhse)
#     end

#     return nothing
# end

