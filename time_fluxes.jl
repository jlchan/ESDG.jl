using ForwardDiff
using BenchmarkTools
using StaticArrays
using FiniteDiff
using LinearAlgebra
using SparseArrays

push!(LOAD_PATH, "./src")
using CommonUtils
using ExplicitJacobians
push!(LOAD_PATH, "./examples/EntropyStableEuler")
using EntropyStableEuler

F(x,y) = (x^2+x*y+y^2)/6
# F(x,y) = 1.0/(x+y)
# F(x,y) = logmean(x,y)
dF(x,y) = ForwardDiff.derivative(y->F(x,y),y)

K = 3
Q = randn(K,K)
Q = Q-transpose(Q)

function ff!(out,u,F,QTr)
    for i = 1:size(QTr,1)
        val = zero(eltype(u))
        ui = u[i]
        for j = 1:size(QTr,2)
            uj = u[j]
            val += QTr[j,i]*F(ui,uj)
        end
        out[i] = val
    end
end

# explicit Jacobian
function jac!(dfdu,u,dF,QTr)
    for i = 1:size(QTr,1)
        ui = u[i]
        for j = 1:size(QTr,2)
            uj = u[j]
            dfdu[i,j] = QTr[j,i]*dF(ui,uj)
        end
    end
    sum_df = sum(dfdu,dims=1)
    for i = 1:size(QTr,1)
        dfdu[i,i] += sum_df[i]
    end
end

u = randn(K)
cache = FiniteDiff.JacobianCache(u)
output = zeros(size(Q))
f!(out,u) = ff!(out,u,F,Q)
function ff(u,Q,F)
    uL,uR = meshgrid(u)
    return sum(Q.*F.(uL,uR),dims=2)
end
f(u) = ff(u,Q,F)

# @btime ForwardDiff.jacobian($f,$u) # AD jacobian
# @btime FiniteDiff.finite_difference_jacobian!($output,$f!,$u,$cache) # FiniteDiff.jl
# @btime jac!($output,$u,$dF,$Q) # explicit Jacobian
# out = similar(u); @btime f!($out,$u) # flux eval
