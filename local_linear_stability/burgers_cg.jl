# my own packages - probably need to add manually
using NodesAndModes # for quadrature
using StartUpDG # for DG setup

# other registered general packages
using UnPack
using LinearAlgebra
using SparseArrays
using Plots
using ForwardDiff

N = 3
K = 10

T = 10.0
CFL = .25
interval = 100

vol_quad = gauss_lobatto_quad(0,0,N)
vol_quad = gauss_quad(0,0,2*N+1)
rd = RefElemData(Line(),N; quad_rule_vol=vol_quad)

VX,EToV = uniform_mesh(Line(),K)
md_BCs = MeshData(VX,EToV,rd)
md = make_periodic(md_BCs,rd)

rd1 = RefElemData(Line(),1)
h = diff(VX)
R = spzeros((N+1)*K,N*K)
sk = 0
for e = 1:K
    global sk
    Np = e==K ? N : N+1
    ids = 1:Np
    R[ids .+ (N+1)*(e-1),ids .+ sk] .= I(Np)
    sk += N
end
R[end,1] = 1

@unpack x,xq,J,rxJ = md
u0(x) = sin(2*pi*(x-.7)) + 2
u0(x) = rand()
uu(x) = u0(x) + 0e-3*cos(pi*x)

@unpack wq,M,Dr,Vq,Vf,Pq = rd
rx = @. rxJ/J
Mg = R'*kron(diagm(J[1,:]),sparse(M))*R
Vxg = kron(diagm(rx[1,:]),sparse(Vq*Dr))*R
wJq = vec(diagm(wq)*(Vq*J))
Vqg = kron(I(K),sparse(Vq))*R

# # weight by u0
# wJq = wJq./(Vqg*(R\u0.(vec(x))))
# Mg = Vqg'*diagm(wJq)*Vqg

u = R\uu.(vec(x)) # CG coeffs
f(u) = u^2/2
# f(u) = u
# fCG(u) = (-Vxg'*(wJq.*f.(Vqg*u)) + Vqg'*((Vqg*u).*wJq.*Vxg*u) )/3 # split form
fCG(u) = -Vxg'*(wJq.*f.(Vqg*u))

# # test advec
# @. f0 *= 0
# f(u) = u

dfdu(u) = ForwardDiff.jacobian(fCG,u)
λ,V = eigen(Mg\dfdu(u))
@show maximum(real(λ))/size(Mg,1)
scatter(real(λ),imag(λ),label="Linearization around u0(x)")
# scatter!(real(λ),imag(λ),label="Linearization around u0(x) + Δu")
plot!(title="Max real part = $(maximum(real(λ)))")
# # Gregor's idea
# x = vec(x)
# v0 = R\(@. u0(x)^2/2)
# v = @. u^2/2
# fCG(v) = -Vxg'*(wJq.*(Vqg*v)) # evars using Gregors trick
# Mg = Vqg'*diagm(wJq.*(Vqg*v0))*Vqg
# dfdu(v) = ForwardDiff.jacobian(fCG,v)
# λ,V = eigen(Mg\dfdu(u))
# scatter(real(λ),imag(λ),label="Linearization around u0(x)")
# Mgv(v) = Vqg'*diagm(wJq.*(Vqg*v))*Vqg
# λ,V = eigen(Mg\dfdu(u))
# scatter!(real(λ),imag(λ),label="Linearization around u0(x) + Δu")
# plot!(title="Max real part = $(maximum(real(λ)))")


# @. u += 1e-3*real.(V[:,end] + V[:,end-1])

# # initial flow rhs
# f0 = Mg\fCG(R\vec(u0.(x).^2/2))
# f0 = 0*(R\vec(x))

# u = R\vec(@. uu(x)^2/2)
# resu = zeros(size(u))
# rk4a,rk4b,rk4c = ck45()
# dt = CFL*(x[2]-x[1]) / maximum(abs.(u))
# Nsteps = ceil(Int,T/dt)
# dt = T/Nsteps
# ubounds = extrema(u)
# unorm = zeros(Nsteps)
# plot()
# @gif for i = 1:Nsteps
#     for INTRK = 1:5
#         rhsu = f0 - Mgv(u)\fCG(u)
#         @. resu = rk4a[INTRK]*resu + dt*rhsu
#         @. u   += rk4b[INTRK]*resu
#     end
#
#     unorm[i] = dot(u,Mg*u)
#
#     if i%interval==0
#         println("$i / $Nsteps: $ubounds, $(extrema(u))")
#         plot(vec(x),R*u,mark=:dot,ms=2,ylims=ubounds .+ (-1,1),leg=false)
#         plot!(title="Step $i out of $Nsteps, final time = $T")
#     end
# end every interval
