using ForwardDiff
using LinearAlgebra
using ToeplitzMatrices
using Plots

N = 100
x = LinRange(-1,1,N+2)[1:end-1]

# fEC(uL,uR) = (1/6)*(uL^2 + uL*uR + uR^2)
# fcentral(uL,uR) = .5*(uL^2/2 + uR^2/2)
# dLF(uL,uR) = .5*max(abs(uL),abs(uR))*(uL-uR)
# ψ(u) = u^3/6

# # # testing advective fluxes
# fEC(uL,uR) = sqrt(uL*uR)
fEC(uL,uR) = logmean(uL,uR)
fcentral(uL,uR) = .5*(uL+uR)
v(u) = log(u)
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


# Q = Circulant([0; -2/3; 1/12; zeros(N+1-5); -1/12; 2/3])
# Q /= h

Q = Circulant([0; -3/4; 3/20; -1/60; zeros(N+1-7); 1/60; -3/20; 3/4])
Q /= h

# an analytical formula for the Jacobian of f(u) = sum(Q.*f(u,u'),dims=2)
compute_rhs(Q,f::F,u) where {F} = 2*sum(Q.*f.(u,u'),dims=2)

function hadsum_mod(Q,u)
    F = @. fEC(u,u')
    for i = 1:length(u)
        for j = i+1:length(u)
            # s = sign(Q[i,j])
            s = 1
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

u0(x) = 2 + sin(pi*x)
u = u0.(x)

CFL = .25
T = .25
interval = 5

resu = zeros(size(u))
rk4a,rk4b,rk4c = ck45()
dt = CFL*h / maximum(abs.(u))
Nsteps = ceil(Int,T/dt)
dt = T/Nsteps
ubounds = extrema(u)
unorm = zeros(Nsteps)
@gif for i = 1:Nsteps
    global u
    for INTRK = 1:5
        # rhsu = vec(b - compute_rhs(Q,fEC,u)) # - sum((@. B*dLF(u,u')),dims=2)
        # rhsu = vec(-hadsum_mod(Q,u) - sum((@. B*dLF(u,u')),dims=2))
        QFu = hadsum_mod(Q,u)
        # QFu = compute_rhs(Q,fEC,u)
        # @show sum(u.*QFu)
        # rhsu = vec(b-QFu)
        rhsu = vec(-QFu)
        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    unorm[i] = norm(u)

    if i%interval==0
        println("$i / $Nsteps: $ubounds, $(extrema(u)), norm(u) = $(unorm[i])")
        # plot(x,u,mark=:dot,ms=2,ylims=ubounds .+ (-.1,.1))
    end
end every interval

# plot(x,u,mark=:dot,ms=2)
plot!(x,(@. u- u0(x-T)),mark=:dot,ms=2,leg=false)
