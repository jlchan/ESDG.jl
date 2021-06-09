using Plots
using OrdinaryDiffEq
using EntropyStableEuler
using SparseArrays, StructArrays, LazyArrays, LinearAlgebra
using Polyester
using ESDG
using TimerOutputs # note: this is the NullOutput branch

N = 3
K1D = 16
CFL = .25
elemType = Quad()
rd = RefElemData(elemType,SBP(),N)
VX,VY,EToV = uniform_mesh(elemType,K1D)
md = MeshData(VX,VY,EToV,rd)
md = make_periodic(md,rd)

# tags for boundary conditions
is_boundary_node = zeros(Int,size(md.xf))
is_boundary_node[md.mapB] .= 1 # change to boundary tag later

function initial_condition(equation::EntropyStableEuler.Euler{2},x,y)
    a,b = .5,.5
    rho = 2.0 + .1*exp(-100*((x-a)^2+(y-b)^2))
    u,v = 0,0
    p = rho^equation.γ
    return prim_to_cons(equation,SVector{4}(rho,u,v,p))
end

function create_cache(equation,rd::RefElemData{Dim,ElemType,SBP},md) where {Dim,ElemType}

    Qr,Qs = (A->rd.M*A).(rd.Drst)
    QrskewTr = -.5*(Qr-Qr')
    QsskewTr = -.5*(Qs-Qs')

    # QrskewTr,QsskewTr = droptol!.(sparse.((QrskewTr,QsskewTr)),1e-13)

    invm = 1 ./ rd.wq

    # tmp variables for entropy projection
    nvars, nvarslog = nvariables(equation), nvariables(equation) + 2
    Np = length(rd.r)
    Uf = StructArray{SVector{nvars,Float64}}(ntuple(_->similar(md.xf),nvars))
    Qlog = StructArray{SVector{nvarslog,Float64}}(undef,Np,md.K) # broadcast matches output type

    rhse = StructArray{SVector{nvars,Float64}}(undef,Np)
    rhse_threads = [similar(rhse) for _ in 1:Threads.nthreads()]

    cache = (;md,QrskewTr,QsskewTr,invm,Fmask=rd.Fmask,wf=rd.wf,
             Uf,Qlog,rhse_threads)

    return cache
end

# preallocate stuff
eqn = EntropyStableEuler.Euler{2}()
@inline two_pt_flux(UL,UR) = fS(EntropyStableEuler.Euler{2}(),UL,UR)
@inline two_pt_flux_logs(QL,QR) = fS_prim_log(EntropyStableEuler.Euler{2}(),QL,QR)
@inline cons_to_prim_logs(u) = cons_to_prim_beta_log(EntropyStableEuler.Euler{2}(),u)

timer = TimerOutput()
# timer = NullTimer()

cache = (;create_cache(eqn,rd,md)..., equation=eqn, is_boundary_node, 
        fS_prim_log = two_pt_flux_logs, fS_EC = two_pt_flux, dissipation=LxF_dissipation, 
        cons_to_prim_logs, solver = nothing, timer) 

function rhs!(dU,U,cache,t)    
    @unpack md = cache
    @unpack rxJ,sxJ,ryJ,syJ,J,nxJ,nyJ,sJ,mapP = md
    @unpack QrskewTr,QsskewTr,invm,Fmask,wf = cache
    @unpack equation, fS_prim_log, fS_EC, dissipation, cons_to_prim_logs, Uf, Qlog = cache
    @unpack timer = cache
    
    @timeit timer "Primitive/log variable transforms" begin
        tmap!(cons_to_prim_logs,Qlog,U)
    end

    @timeit timer "Extract face vals" begin
        @batch for e = 1:size(U,2)
            for (i,vol_id) in enumerate(Fmask)        
                Uf[i,e] = U[vol_id,e]        
            end
        end
    end

    @batch for e = 1:md.K

        @timeit timer "Volume terms" begin        
            rhse = cache.rhse_threads[Threads.threadid()]
            fill!(rhse,zero(eltype(rhse)))

            QxTr = LazyArray(@~ @. 2 * (rxJ[1,e]*QrskewTr + sxJ[1,e]*QsskewTr))
            QyTr = LazyArray(@~ @. 2 * (ryJ[1,e]*QrskewTr + syJ[1,e]*QsskewTr))           
            hadamard_sum_ATr!(rhse, (QxTr,QyTr), fS_prim_log, view(Qlog,:,e)) 

            # Grx = LazyArray(@~ @. rxJ[:,e]+rxJ[:,e]')
            # Gry = LazyArray(@~ @. ryJ[:,e]+ryJ[:,e]')
            # Gsx = LazyArray(@~ @. sxJ[:,e]+sxJ[:,e]')
            # Gsy = LazyArray(@~ @. syJ[:,e]+syJ[:,e]')            
            # hadamard_sum_ATr!(rhse, (QrskewTr,QrskewTr), (Grx,Gry), fS_prim_log, view(Qlog,:,e))
            # hadamard_sum_ATr!(rhse, (QsskewTr,QsskewTr), (Gsx,Gsy), fS_prim_log, view(Qlog,:,e))

            # @infiltrate
        end

        @timeit timer "Interface terms" begin
            for (i,vol_id) = enumerate(Fmask)
                UM = Uf[i,e]
                normal = SVector{2}(nxJ[i,e], nyJ[i,e]) / sJ[i,e]
                UP = Uf[mapP[i,e]]                
                Fx,Fy = fS_EC(UP,UM)
                diss = dissipation(equation,normal,UM,UP)
                val = (Fx * nxJ[i,e] + Fy * nyJ[i,e] + diss*sJ[i,e]) * wf[i]
                rhse[vol_id] += val
            end
        end

        @timeit timer "Store output" begin
            @. rhse = -rhse / J[1,e] # split up broadcasts to avoid allocations
            @. dU[:,e] = invm * rhse             
        end
    end

    return nothing
end

U = StructArray{SVector{nvariables(eqn),Float64}}(undef,size(md.x)...)
U .= ((x,y)->initial_condition(eqn,x,y)).(md.xyz...)

# dU = similar(U)
# rhs!(dU,U,cache,0.0)

dt0 = CFL * estimate_h(rd,md) / inverse_trace_constant(rd)
tspan = (0.0,1.0)
ode = ODEProblem(rhs!,U,tspan,cache)
sol = solve(ode, SSPRK43(), dt=dt0, save_everystep=false, callback=monitor_callback())
show(cache.timer)

U = sol.u[end]
plot(DGTriPseudocolor(StructArrays.component(U,1),rd,md),ratio=1,color=:blues)
plot!(MeshPlotter(rd,md),linecolor=:white,lw=.5)