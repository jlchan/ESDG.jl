using Plots
using OrdinaryDiffEq
using ESDG
using EntropyStableEuler
using StructArrays,LazyArrays,LinearAlgebra
# using Octavian

rd = RefElemData(Tri(), N=3)
triout = square_hole_domain()
VX,VY,EToV = triangulateIO_to_VXYEToV(triout)
md = MeshData(VX,VY,EToV,rd)

# make tags for boundary conditions
is_boundary_node = zeros(Int,size(md.xf))
is_boundary_node[md.mapB] .= 1 # change to boundary tag later
# is_boundary_node = get_node_boundary_tags(triout,rd,md)

function initial_condition(x,y)
    rho = 2.0
    u,v = 0,0
    p = 1.0
    return prim_to_cons(EntropyStableEuler.Euler{2}(),SVector{4}(rho,u,v,p))
end

U = StructArray{SVector{4,Float64}}(undef,size(md.x)...)
U .= initial_condition.(md.xyz...)

function create_rhs_cache(equation,rd::RefElemData{Dim},md) where {Dim}

    Qrh,Qsh,VhP,Ph = hybridized_SBP_operators(rd)
    QrhskewTr,QshskewTr = (A->typeof(A)(-.5*(A-A'))).((Qrh,Qsh))

    project_and_store! = let Ph=Ph
        (y,x)->mul!(y,Ph,x) # can't use matmul! b/c its applied to a subarray
    end

    # tmp variables for entropy projection
    nvars, nvarslog = nvariables(equation), nvariables(equation) + Dim
    Uq = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xq),nvars)) # cons vars at quad pts
    Uh = StructArray{SVector{nvars,Float64}}(ntuple(_->similar([md.xq;md.xf]),nvars)) # entropy vars at hybridized pts
    VUq, VUh = similar(Uq), similar(Uh)  # entropy vars at quad pts
    Qhlog = StructArray{SVector{nvarslog,Float64}}(undef,size(Uh)) # broadcast matches output type

    rhse_threads = [similar(Uh[:,1]) for _ in 1:Threads.nthreads()]

    rhs_cache = (; rd, md, 
            QrhskewTr, QshskewTr, VhP, Ph,
            Uq, VUq, VUh, Uh, Qhlog,
            project_and_store!,
            rhse_threads)

    return rhs_cache
end

function compute_entropy_projection!(U,rhs_cache,equation,solver)
    @unpack rd = rhs_cache
    @unpack VhP, Uq, VUq, VUh, Uh, Qhlog = rhs_cache

    StructArrays.foreachfield((uout,u)->matmul!(uout,rd.Vq,u),Uq,U)
    tmap!(u->cons_to_entropy(equation,u),VUq,Uq) 
    StructArrays.foreachfield((uout,u)->matmul!(uout,VhP,u),VUh,VUq)
    tmap!(v->entropy_to_cons(equation,v),Uh,VUh) 

    # convert to primitive log variables 
    tmap!(u->cons_to_prim_beta_log(equation,u),Qhlog,Uh)

    # extract face values
    Nh,Nq = size(VhP)
    Uf = view(Uh,Nq+1:Nh,:)

    return Qhlog,Uf
end

# # preallocate stuff
eqn = EntropyStableEuler.Euler{2}()
dU = similar(U)
@inline two_pt_flux(UL,UR) = fS(EntropyStableEuler.Euler{2}(),UL,UR)
@inline two_pt_flux_logs(QL,QR) = fS_prim_log(EntropyStableEuler.Euler{2}(),QL,QR)
rhs_cache = (;create_rhs_cache(eqn,rd,md)..., equations=eqn, is_boundary_node, 
             fS_prim_log = two_pt_flux_logs, fS = two_pt_flux, dissipation=LxF_dissipation,
             solver = nothing) 


function rhs!(dU,U,rhs_cache,t)
    @unpack rd,md,equations = rhs_cache
    @unpack wf = rd
    @unpack rxJ,sxJ,ryJ,syJ,J = md
    @unpack nxJ,nyJ,sJ,mapP = md
    @unpack QrhskewTr,QshskewTr,VhP = rhs_cache
    @unpack project_and_store! = rhs_cache
    @unpack solver = rhs_cache
    @unpack fS_prim_log, fS, dissipation = rhs_cache

    Nh,Nq = size(VhP)
    skip_index = let Nq=Nq
        (i,j) -> i > Nq && j > Nq
    end
    
    Qhlog,Uf = compute_entropy_projection!(U,rhs_cache,equations,solver) 

    @batch for e = 1:md.K
        rhse = rhs_cache.rhse_threads[Threads.threadid()]

        fill!(rhse,zero(eltype(rhse)))
        Qe = view(Qhlog,:,e)   
        QxTr = LazyArray(@~ @. 2 *(rxJ[1,e]*QrhskewTr + sxJ[1,e]*QshskewTr))
        QyTr = LazyArray(@~ @. 2 *(ryJ[1,e]*QrhskewTr + syJ[1,e]*QshskewTr))

        hadamard_sum_ATr!(rhse, (QxTr,QyTr), fS_prim_log, Qe, skip_index) 

        for (i,vol_id) = enumerate(Nq+1:Nh)
            if rhs_cache.is_boundary_node[i,e] == 1
                UM = Uf[i,e]
                ϱ, ϱu, ϱv, E = UM
                u, v = ϱu/ϱ, ϱv/ϱ
                nx, ny = nxJ[i,e]/sJ[i,e], nyJ[i,e]/sJ[i,e]
                u_n = u*nx + v*ny
                uP = u - 2*u_n*nx
                vP = v - 2*u_n*ny
                UP = SVector{4}(ϱ,ϱ*uP,ϱ*vP,E) 

                Fx,Fy = fS(UP,UM)
                normal = SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e]
                diss = dissipation(equations,normal,UM,UP)
                val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
            else
                UM, UP = Uf[i,e], Uf[mapP[i,e]]
                Fx,Fy = fS(UP,UM)
                normal = SVector{2}(nxJ[i,e],nyJ[i,e])/sJ[i,e]
                diss = dissipation(equations,normal,UM,UP)
                val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
            end            
            rhse[vol_id] += val
        end

        @. rhse = -rhse / J[1,e]

        # project down and store
        StructArrays.foreachfield(project_and_store!, view(dU,:,e), rhse)
    end

    return nothing
end

dt0 = CFL * estimate_h(rd,md) / inverse_trace_constant(rd)
tspan = (0.0,.5)
ode = ODEProblem(rhs!,U,tspan,cache)
sol = solve(ode, SSPRK43(), dt=dt0, save_everystep=false, callback=monitor_callback())
show(cache.timer)