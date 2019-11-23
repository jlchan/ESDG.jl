using BenchmarkTools
using StaticArrays

N = 2
u,v = ntuple(x->randn(N),2)

function foo!(u,v)
    u .= v
end

@show u
foo!(u,v)
@show u

# foo1(U) = (x->x[:,1]).(U)
# foo2(U) = tuple((x->x[:,1]).(U)...)
# @btime foo1($U)
# @btime foo2($U)

# N = 10000
# a,b,c,d = ntuple(x->randn(N),4)
# X = (a,b)
# Y = (c,d)
# X = [a,b]
# Y = [c,d]

# X = SizedVector{2}(a,b)
# Y = SizedVector{2}(c,d)

#
# function foo(a,b,c,d)
#     ab = exp(a*b)
#     return ab+c, ab-d
# end
#
# function foo_loop(N,a,b,c,d)
#     for i = 1:N
#         foo(a[i],b[i],c[i],d[i])
#     end
# end
# @btime foo_loop($N,$a,$b,$c,$d)
#
# function foo_loop_splat!(N,X,Y)
#     for i = 1:N
#         foo(getindex.(X,i)...,getindex.(Y,i)...)
#     end
# end
# @btime foo_loop_splat!($N,$X,$Y)
#
# # replace tuples with sized vectors
# @btime foo_loop_splat!($N,$X,$Y)
