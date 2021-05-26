using OrdinaryDiffEq
using UnPack

function lorenz!(du,u,p,t)
    @unpack σ,β,δ = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(β-u[3]) - u[2]
    du[3] = u[1]*u[2] - δ*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,100.0)
p = (; σ=10, β=28, δ=8/3)
prob = ODEProblem(lorenz!,u0,tspan,p)

# taken from https://github.com/trixi-framework/Trixi.jl/blob/99a7e2840872d0f1180a984dc034ec81be462a0b/src/callbacks_step/callbacks_step.jl#L17
@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
        isempty(integrator.opts.tstops) ||
        integrator.iter == integrator.opts.maxiters
end

default_monitor(integrator) = println("on timestep $(integrator.iter), at time $(integrator.t).")

"""
    function monitoring_callback(interval=10, monitor=default_monitor)

See https://diffeq.sciml.ai/stable/basics/integrator/#Handing-Integrators for `integrator` fields. 
The default monitoring function just prints out timestep and current time. 
"""
function monitoring_callback(interval=10, monitor=default_monitor)
    condition(u,t,integrator) = integrator.iter % interval ==0 || isfinished(integrator)
    return DiscreteCallback(condition, monitor, save_positions=(false,false))
end
cb = monitoring_callback(10)

# can't believe this just works.
sol = solve(prob,Tsit5(),dt = .1,save_everystep=false, callback=cb)
