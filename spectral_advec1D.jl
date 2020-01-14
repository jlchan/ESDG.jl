using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
N   = 15
CFL = .75
T   = 10 # endtime

"Construct matrices on reference elements"
x = sort(@. cos(pi*(0:N)/N))
V = vandermonde_1D(N, x)
Dr = grad_vandermonde_1D(N, x)/V
B = zeros(N+1,2)
B[1,1] = -1
B[N+1,2] = 1
B = (N+1)^2/2 * B  # magic scaling!
# B = @. 2/(N*(N+1))*B

"initial conditions"
u0(x) = @. exp(-25*x^2)
u0(x) = @. sin(2*pi*x)
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
plt = plot(Vp*x,Vp*u,ylims=ulims,title="Timestep 0 out of $Nsteps")

resu = zeros(size(x))
@gif for i = 1:Nsteps
    for INTRK = 1:5
        uL = u[[end,1]]
        uR = u[[1,end]]
        tau = 0
        rhsu = Dr*u - B*((uR-uL)/2)
        @. rhsu = -rhsu

        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    if i%100==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        # display(plot(Vp*x,Vp*u,ylims=(-.1,1.1)))
        # push!(plt, Vp*x,Vp*u,ylims=(-.1,1.1))
        plot(Vp*x,Vp*u,ylims=ulims,title="Timestep $i out of $Nsteps")
        scatter!(x,u)
        # sleep(.0)
    end
end every 100

# plot(Vp*x,Vp*u-u0(Vp*x))
# scatter(x,u,ylims=ulims)
# plot!(Vp*x,u0(Vp*x),ylims=ulims)
