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
#function resetCheapThreads()
function resetThreads()
    CheapThreads.reset_workers!()
    # Polyester.reset_workers!()
    ThreadingUtilities.reinitialize_tasks!()
end
