using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 24
T   = 10 # endtime
CFL = .75

"Construct matrices on reference elements"
x,w = gauss_lobatto_quad(0,0,N)
V   = vandermonde_1D(N, x)
D   = grad_vandermonde_1D(N, x)/V # D = inv(M)*Q
B   = zeros(N+1,2)
B[1,1] = 1
B[N+1,2] = 1
invM = V*V'
L = invM*B

"initial conditions"
u0(x) = @. exp(-25*x^2)
# u0(x) = @. sin(2*pi*x)
u0(x) = Float64.(@. abs(x) < .33)

u = u0(x)

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
h = minimum(diff(x))
dt = CFL * h
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"plotting nodes"
xp = LinRange(-1,1,1000)
Vp = vandermonde_1D(N,xp)/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=4)
ulims = (minimum(u)-.5,maximum(u)+.5)

plot(xp,Vp*u,ylims=ulims,title="Timestep 0 out of $Nsteps")
scatter!(x,u)

resu = zeros(size(x))
@gif for i = 1:Nsteps
    for INTRK = 1:5
        uL = u[[end,1]] # [u_N+1, u_1]
        uR = u[[1,end]] # [u_1, u_N+1]
        rhsu = D*u - L*([-1;1].*(uR-uL)/2)
        @. rhsu = -rhsu

        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        plot(xp,Vp*u,ylims=ulims,title="Timestep $i out of $Nsteps")
        scatter!(x,u)
    end
end every 100

# "compare computed + exact solutions"
scatter(x,u,ylims=ulims)
plot!(xp,u0(Vp*x),ylims=ulims)

# plot(xp,Vp*u-u0(Vp*x)) # plot error
