using LinearAlgebra
using SparseArrays
using Plots

N = 10
Q = spdiagm(1=>ones(N-1),-1=>-ones(N-1))
Q[1,end] = -1
Q[end,1] = 1

xv = LinRange(-1,1,N+1)
h = xv[2]-xv[1]
xref = xv[1:end-1] .+ h/2

x(x̂,t) = 
dxdt(x̂,t) = .1*sin(pi*x̂)*sin(pi*t) 


# scatter(x̂,0*x̂)
# scatter!(x̂v,0*x̂v,marker=:square)