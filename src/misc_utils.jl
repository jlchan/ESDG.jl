
"""
    function tmap!(f,out,x)

Threaded `map!` function using Polyester.@batch.
"""
@inline function tmap!(f,out,x)
    @batch for i in eachindex(x)
        @inbounds out[i] = f(x[i])
    end
    return out # good practice for mutating functions?
end

"""
    function resetCheapThreads()

If CheapThreads freezes, running this might fix it. Must run manually (not sure how to automatically detect freezes). 
"""
function resetThreads()
    Polyester.reset_workers!()
    ThreadingUtilities.reinitialize_tasks!()
end
