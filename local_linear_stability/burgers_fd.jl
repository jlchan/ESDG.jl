using ForwardDiff
using LinearAlgebra
using ToeplitzMatrices
using Plots

N = 100
x = LinRange(-1,1,N+2)[1:end-1]
h = x[2] - x[1]

fEC(uL,uR) = (1/6)*(uL^2 + uL*uR + uR^2)
fcentral(uL,uR) = .5*(uL^2/2 + uR^2/2)
dLF(uL,uR) = .5*max(abs(uL),abs(uR))*(uL-uR)
f(u) = u^2/2
v(u) = u

# # # testing advective fluxes
# fEC(uL,uR) = sqrt(uL*uR)
# fEC(uL,uR) = logmean(uL,uR)
# fcentral(uL,uR) = .5*(uL+uR)
# v(u) = log(u)
# f(u) = u
# # # dLF(uL,uR) = .5*(uL-uR)
# # # (uL-uR)*(uL+uR)/2 = uL^2/2 - uR^2/2

dfEC(uL,uR) = ForwardDiff.derivative(uR->fEC(uL,uR),uR)
dfcentral(uL,uR) = ForwardDiff.derivative(uR->fcentral(uL,uR),uR)
ddLF(uL,uR) = ForwardDiff.derivative(uR->dLF(uL,uR),uR)

# make skew matrix - 2nd order FD matrix
Q = diagm(1=>ones(N),-1=>-ones(N))
Q[1,end] = -1
Q[end,1] = 1
@. Q *= .5/h
# h = 2/N
h = x[2]-x[1]

B = abs.(Q)/h

Q = Circulant([0; -2/3; 1/12; zeros(N+1-5); -1/12; 2/3])
Q /= h

Q = Circulant([0; -3/4; 3/20; -1/60; zeros(N+1-7); 1/60; -3/20; 3/4])
Q /= h

# an analytical formula for the Jacobian of f(u) = sum(Q.*f(u,u'),dims=2)
compute_rhs(Q,f::F,u) where {F} = 2*sum(Q.*f.(u,u'),dims=2)
compute_jac_AD(Q,df::F) where {F} = ForwardDiff.jacobian(u->compute_rhs(Q,f,u),u)

function compute_jac(Q,df::F,u) where {F}
    jac = @. Q*df.(u,u')
    jac -= diagm(vec(sum(jac,dims=1)))
    return -jac
end

function hadsum_mod(Q,u)
    F = @. fEC(u,u')
    for i = 1:length(u)
        for j = i+1:length(u)
            s = sign(Q[i,j])
            # s = 1
            fc = fcentral(u[i],u[j])
            Δv = v(u[i])-v(u[j])
            if s*Δv*(fc-F[i,j]) > 0
                F[i,j] += fc - F[i,j]
                F[j,i] += fc - F[i,j]
            else
                F[i,j] += F[i,j]-fc
                F[j,i] += F[j,i]-fc
            end
        end
    end
    return 2*sum(Q.*F,dims=2)
end

function hadsum_mod_smooth(Q,u,p,tol)
    F = @. fEC(u,u')
    for i = 1:length(u)
        for j = i+1:length(u)
            s = sign(Q[i,j])
            Δv = v(u[i])-v(u[j])
            fc = fcentral(u[i],u[j])
            delta = s*Δv*(fc-F[i,j])

            beta = abs(delta^p + tol)^(1/p)
            alpha = (beta - delta)/(2*beta)

            F[i,j] = alpha*fc + (1-alpha)*F[i,j]
            F[j,i] = alpha*fc + (1-alpha)*F[j,i]
        end
    end
    return 2*sum(Q.*F,dims=2)
end

# baseflow for Burgers' and linearization
u0(x) = 2 + sin(pi*(x-.7))
u = @. u0(x) + 1e-3*cos(pi*x)
# u = rand(N+1)

# λcentral = eigvals(compute_jac(Q,dfcentral,u))
# scatter(λcentral,label="Central")
# λEC = eigvals(compute_jac(Q,dfEC,u))
# scatter!(λEC,label="ES",ms=2)
# @show maximum(real.(λcentral))
# dfdu_mod = ForwardDiff.jacobian(u->-hadsum_mod(Q,u),u)
# λ = eigvals(dfdu_mod)
# scatter!(λ,label="Mod",mark=:square,ms=3)
# plot!(title="Max real mod λ = $(maximum(real.(λ)))")

# code to test dissipation terms too
function compute_jac(Q,df::F1,K,dd::F2) where {F1,F2}
    jac = @. Q*df(u,u') + K*dd(u,u')
    jac -= diagm(vec(sum(jac,dims=1)))
    return -jac
end

# scatter(eigvals(compute_jac(Q,dfcentral,abs.(B),ddLF)),label="Cen+LF")
# scatter!(eigvals(compute_jac(Q,dfEC,abs.(B),ddLF)),label="ES+LF",ms=2)
# scatter!(λ,label="Mod 2",mark=:square,ms=3)
# plot!(title="Max real mod λ = $(maximum(real.(λ)))",legend=:topleft)


uu(x) = 1-.9*sin(pi*x)
u = uu.(x)
dfdx_exact(x) = ForwardDiff.derivative(x->uu(x)^2/2,x)

aa(uL,uR) = avg(uL,uR)/sqrt(uL^2+uR^2)
ff(uL,uR) = aa(uL,uR)*fcentral(uL,uR) + (1-aa(uL,uR))*fEC(uL,uR)
println("===== h = $h =====")
@show norm(dfdx_exact.(x) - Q*f.(u))
@show norm(dfdx_exact.(x) - compute_rhs(Q,fEC,u))
# @show norm(Q*f.(u) - sum((@. 2*Q*ff(u,u')),dims=2))
# @show norm(Q*f.(u) - hadsum_mod(Q,u))
tol = 1e-4
@show norm(dfdx_exact.(x) - hadsum_mod_smooth(Q,u,2,tol))
@show norm(dfdx_exact.(x) - hadsum_mod_smooth(Q,u,4,tol^2))
# CFL = .25
# T = 1.0
# interval = 5
# b = compute_rhs(Q,fcentral,u0.(x))
#
# resu = zeros(size(u))
# rk4a,rk4b,rk4c = ck45()
# dt = CFL*h / maximum(abs.(u))
# Nsteps = ceil(Int,T/dt)
# dt = T/Nsteps
# ubounds = extrema(u)
# unorm = zeros(Nsteps)
# @gif for i = 1:Nsteps
#     global u
#     for INTRK = 1:5
#         # rhsu = vec(b - compute_rhs(Q,fEC,u)) # - sum((@. B*dLF(u,u')),dims=2)
#         # rhsu = vec(-hadsum_mod(Q,u) - sum((@. B*dLF(u,u')),dims=2))
#         # QFu = hadsum_mod(Q,u)
#         QFu = compute_rhs(Q,fcentral,u)
#         # # @show sum(u.*QFu)
#         # rhsu = vec(b-QFu)
#         rhsu = vec(-QFu)
#         @. resu = rk4a[INTRK]*resu + dt*rhsu
#         @. u   += rk4b[INTRK]*resu
#     end
#
#     unorm[i] = norm(u)
#
#     if i%interval==0
#         println("$i / $Nsteps: $ubounds, $(extrema(u)), norm(u) = $(unorm[i])")
#         # plot(x,u,mark=:dot,ms=2,ylims=ubounds .+ (-.1,.1))
#     end
# end every interval
#
# plot!(x,u,mark=:dot,ms=2,ylims=ubounds)
