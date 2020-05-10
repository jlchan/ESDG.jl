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

u = 1 .+ rand(K)
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

function LF(uL,uR)
        return (@. .5*max(abs(uL),abs(uR))*(uL-uR))
        # return (@. (uL-uR))
end
B = randn(K,K)
B = B'*B
# B = [0 1;1 0]
d!(out,u) = ff!(out,u,LF,B)
FiniteDiff.finite_difference_jacobian!(output,d!,u,cache)

uR,uL = meshgrid(u)
dxLF(uL,uR) = ForwardDiff.derivative(uL->LF(uL,uR),uL)
dyLF(uL,uR) = ForwardDiff.derivative(uR->LF(uL,uR),uR)
jacx = -B.*transpose(dxLF.(uL,uR)) + diagm(vec(sum(B.*transpose(dxLF.(uL,uR)),dims=1)))
jacy = B.*dyLF.(uL,uR) - diagm(vec(sum(B.*dyLF.(uL,uR),dims=1)))

dxLFJ(uL,uR) = ForwardDiff.jacobian(uL->LF(uL,uR),uL)
dyLFJ(uL,uR) = ForwardDiff.jacobian(uR->LF(uL,uR),uR)
jacx2 = Matrix(hadamard_jacobian(sparse(B), dxLFJ, [u]))
jacy2 = Matrix(hadamard_jacobian(sparse(B), dyLFJ, [u]))

@show norm(output - jacx), norm(jacx - jacx2)
@show norm(output - jacy), norm(jacy - jacy2)

# F(x,y)  = SVector{2}(x[1]*y[2],y[1]+x[2])
# dF(x,y) = ForwardDiff.jacobian(y->F(x,y),y)
# uL = @SVector [.1 .+ rand(10000),.1 .+ rand(10000)]
# uR = @SVector [.1 .+ rand(10000),.1 .+ rand(10000)]
#
# function feval(uL,uR,F)
#     for i = 1:length(uL)
#         uLi = getindex.(uL,i)
#         uRi = getindex.(uR,i)
#         F(uLi,uRi)
#     end
# end
#
# @btime feval($uL,$uR,$F)
# @btime feval($uL,$uR,$dF)
