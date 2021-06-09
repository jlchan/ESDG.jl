using Plots
using OrdinaryDiffEq
using EntropyStableEuler
using StructArrays, LazyArrays, LinearAlgebra
using Polyester
using ESDG
using TimerOutputs # note: this is the NullOutput branch

N = 3
K1D = 16
CFL = .5
elemType = Quad()
rd = RefElemData(elemType,SBP(),N)
VX,VY,EToV = uniform_mesh(elemType,K1D)
md = MeshData(VX,VY,EToV,rd)
# md = make_periodic(md,rd)

# tags for boundary conditions
is_boundary_node = zeros(Int,size(md.xf))
is_boundary_node[md.mapB] .= 1 # change to boundary tag later



# get face centroids
function coordinate_face_centroids(xf)
    Nfp = size(md.xf,1)÷rd.Nfaces
    xc = reshape(xf,Nfp,rd.Nfaces*md.K)
    return vec(typeof(xf)(sum(xc,dims=1)/size(xc,1)))
end
compute_face_centroids(md) = map(coordinate_face_centroids,md.xyzf)
xc,yc = compute_face_centroids(md)
boundary_face_ids = findall(vec(md.FToF) .== 1:length(md.FToF))
xb,yb = map(x->x[boundary_face_ids],(xc,yc))

# tag = 
# tagged_boundary_faces = boundary_face_ids[tag]

# is_boundary_face = zeros(Int,rd.Nfaces*md.K)
# is_boundary_face[boundary_face_ids] .= 1

# Nfp = size(md.xf,1)
# is_boundary_node = reshape(repeat(is_boundary_face',Nfp,1),Nfp*rd.Nfaces,md.K)




# struct WallBC end
# struct InflowBC{N,Tv}
#     U_inflow::SVector{N,Tv}
# end
# struct PressureOutflowBC{Tv}
#     p_outflow::Tv
# end
# bcDict = Dict(1=>WallBC(),2=>InflowBC())

# function get_BC_state(bc::WallBC,equation::EntropyStableEuler.Euler{2},UM,normal,xyzt...)
#     return wall_boundary_state(equation,normal,UM)
# end
# function get_BC_state(bc::InflowBC,equation::EntropyStableEuler.Euler{2},UM,normal,xyzt...)
#     return SVector{d+2}(bc(equation,xyzt...))
# end
# function get_BC_state(bc::PressureOutflow,equation::EntropyStableEuler.Euler{2},UM,normal,xyzt...)
#     ρ,ρu,ρv,_ = UM
#     rho_unorm2 = (ρu^2 + ρv^2)/ρ
#     E_out = outflow_pressure / (equation.γ-1) + .5*rho_unorm2
#     return SVector{d+2}(ρ,ρu,ρv,E_out)
# end

function Dirichlet_boundary_state(equation::EntropyStableEuler.Euler{d},xyzt...) where {d}
    return SVector{d+2}(U_function(xyzt...))
end

function pressure_outflow_state(equation::EntropyStableEuler.Euler{d},outflow_pressure,UM) where {d}
    ρ,ρu,ρv,_ = UM
    rho_unorm2 = (ρu^2 + ρv^2)/ρ
    E_out = outflow_pressure / (equation.γ-1) + .5*rho_unorm2
    return SVector{d+2}(ρ,ρu,ρv,E_out)
end

function wall_boundary_state(equation::EntropyStableEuler.Euler{2},normal,UM)
    ϱ, ϱu, ϱv, E = UM
    u, v = ϱu/ϱ, ϱv/ϱ
    nx, ny = normal
    u_n = u*nx + v*ny
    uP = u - 2*u_n*nx
    vP = v - 2*u_n*ny
    return SVector{4}(ϱ,ϱ*uP,ϱ*vP,E) 
end

function initial_condition(equation::EntropyStableEuler.Euler{2},x,y)
    a,b = .5,.5
    rho = 2.0 + .5*exp(-100*((x-a)^2+(y-b)^2))
    u,v = 0,0
    p = rho^equation.γ
    return prim_to_cons(equation,SVector{4}(rho,u,v,p))
end

function create_cache(equation,rd::RefElemData{Dim,ElemType,SBP},md) where {Dim,ElemType}

    Qr,Qs = (A->rd.M*A).(rd.Drst)
    QrskewTr = -.5*(Qr-Qr')
    QsskewTr = -.5*(Qs-Qs')
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
timer = NullTimer()

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
            # hadamard_sum_ATr!(rhse, (QxTr,QyTr), fS, view(U,:,e)) 
        end

        @timeit timer "Interface terms" begin
            for (i,vol_id) = enumerate(Fmask)
                UM = Uf[i,e]
                normal = SVector{2}(nxJ[i,e], nyJ[i,e]) / sJ[i,e]
                if cache.is_boundary_node[i,e] == 1
                    UP = wall_boundary_state(equation,normal,UM)
                else
                    UP = Uf[mapP[i,e]]
                end            
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
dU = similar(U)

dt0 = CFL * estimate_h(rd,md) / inverse_trace_constant(rd)
tspan = (0.0,1.0)
ode = ODEProblem(rhs!,U,tspan,cache)
sol = solve(ode, RDPK3SpFSAL35(), dt=dt0, save_everystep=false, callback=monitor_callback())
show(cache.timer)

U = sol.u[end]
plot(DGTriPseudocolor(StructArrays.component(U,1),rd,md),ratio=1,color=:blues)
plot!(MeshPlotter(rd,md),linecolor=:white,lw=.5)