default_monitor(integrator) = println("on timestep $(integrator.iter), at time $(integrator.t).")

"""
    function monitoring_callback(interval=10, monitor=default_monitor)

Returns a callback to monitor the progress of an ODE solver given a function `monitor(integrator)`. 
Defaults to `default_monitor`, which just prints out the timestep and current time. 
See https://diffeq.sciml.ai/stable/basics/integrator/#Handing-Integrators for `integrator` fields. 

WARNING - don't modify any `integrator` fields within `monitor` or it may ruin the ODE solve.
"""
function monitor_callback(interval=10, monitor=default_monitor)
    condition(u,t,integrator) = (interval > 0) && (integrator.iter % interval ==0 || isfinished(integrator))
    return DiscreteCallback(condition, monitor, save_positions=(false,false))
end


# copied from https://github.com/trixi-framework/Trixi.jl/blob/99a7e2840872d0f1180a984dc034ec81be462a0b/src/callbacks_step/callbacks_step.jl#L17-L23
@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
        isempty(integrator.opts.tstops) ||
        integrator.iter == integrator.opts.maxiters
end

# TODO: add SavingCallback given a QOI functional https://tutorials.sciml.ai/html/introduction/04-callbacks_and_events.html
