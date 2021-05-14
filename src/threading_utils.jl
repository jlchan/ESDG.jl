# =================== threading utilities ===================
@inline function tmap!(f,out,x)
    Trixi.@threaded for i in eachindex(x)
        @inbounds out[i] = f(x[i])
    end
end

## workaround for matmul! with threads https://discourse.julialang.org/t/odd-benchmarktools-timings-using-threads-and-octavian/59838/5
@inline function bmap!(f,out,x)
    @batch for i in eachindex(x)
            @inbounds out[i] = f(x[i])
    end
end