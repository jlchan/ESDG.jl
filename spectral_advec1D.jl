using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 50
CFL = .75
T   = 10 # endtime

"Construct matrices on reference elements"
x,w = gauss_lobatto_quad(0,0,N)
V = vandermonde_1D(N, x)
Dr = grad_vandermonde_1D(N, x)/V
B = zeros(N+1,2)
B[1,1] = 1
B[N+1,2] = 1
invM = V*V'
L = invM*B

"initial conditions"
u0(x) = @. exp(-100*x^2)
# u0(x) = @. sin(2*pi*x)
# u0(x) = Float64.(@. abs(x) < .5)

u = u0(x)

"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
h = minimum(diff(x))
dt = CFL * h
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"plotting nodes"
Vp = vandermonde_1D(N,LinRange(-1,1,100))/V
gr(size=(300,300),legend=false,markerstrokewidth=1,markersize=2)
ulims = (minimum(u)-.5,maximum(u)+.5)

resu = zeros(size(x))
# @gif
for i = 1:Nsteps
    for INTRK = 1:5
        uL = u[[end,1]]
        uR = u[[1,end]]
        tau = 0
        rhsu = Dr*u - L*([-1;1].*(uR-uL)/2)
        @. rhsu = -rhsu

        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        plot(Vp*x,Vp*u,ylims=ulims,title="Timestep $i out of $Nsteps")
        scatter!(x,u)
    end
end #every 100

# "compare computed + exact solutions"
scatter(x,u,ylims=ulims)
plot!(Vp*x,u0(Vp*x),ylims=ulims)

# plot(Vp*x,Vp*u-u0(Vp*x)) # plot error
