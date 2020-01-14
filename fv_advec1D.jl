using Revise # reduce need for recompile
using Plots
using LinearAlgebra

push!(LOAD_PATH, "./src") # user defined modules
using CommonUtils, Basis1D

"Approximation parameters"
K   = 1024 # number of elements
CFL = 1   # controls size of timestep
T   = 10  # endtime

"Grid related variables"
VX = LinRange(-1,1,K+1)
EToV = repeat([0 1],K,1) + repeat(1:K,1,2) #transpose(reshape(sort([1:K; 2:K+1]),2,K))
h = diff(VX) # mesh spacing
x = mean(VX[EToV],2) # cell centers

"initial conditions"
u0(x) = @. exp(-25*x^2)
# u0(x) = @. sin(2*pi*x)
# u0(x) = Float64.(@. abs(x) < .5)
u = u0(x)



"Time integration"
rk4a,rk4b,rk4c = rk45_coeffs()
dt = CFL * 2 / K
Nsteps = convert(Int,ceil(T/dt))
dt = T/Nsteps

"plotting nodes"
gr(size=(300,300),legend=false,markerstrokewidth=0,markersize=2,linewidth=2)
# l = @layout [a{0.5w} b]
ulims = (minimum(u)-.5,maximum(u)+.5)
plt = scatter(x,u,ylims=ulims,title="Timestep 0 out of $Nsteps")

resu = zeros(size(x))
@gif for i = 1:Nsteps
    for INTRK = 1:5
        uL = vcat(u[end],u)
        uR = vcat(u,u[1])
        # flux = @. .5*(uL+uR) # central flux
        # flux = @. uL
        tau = .25
        flux = @. .5*(uL+uR) - tau*.5*(uR-uL)
        rhsu = -diff(flux,dims=1)./h

        @. resu = rk4a[INTRK]*resu + dt*rhsu
        @. u   += rk4b[INTRK]*resu
    end

    if i%25==0 || i==Nsteps
        println("Number of time steps $i out of $Nsteps")
        scatter(x,u,ylims=ulims,title="Timestep $i out of $Nsteps")
    end
end every 25

plot(x,u,ylims=ulims,linestyle=:dash)
plot!(x,u0(x),ylims=ulims)
