# function counter()
#     n = 0
#     ()->n+=1
# end
#
# addOne = counter()
# addOneAgain = counter()
#
# @show addOne()
# @show addOne()
# @show addOneAgain()

using BenchmarkTools

N = 3
X = (zeros(N),zeros(N))
Y = [1,2]
function foo1!(X,Y,i)
    for fld in eachindex(X)
        X[fld][i] = Y[fld]
    end
end
@btime foo1!($X,$Y,$1)

foo2!(x,y,i) = x[i] = y
@btime foo2!.($X,$Y,$1)

foo3!(X,Y,i) = getindex(X,i) = Y
@btime foo3!($X,$Y,$1)
