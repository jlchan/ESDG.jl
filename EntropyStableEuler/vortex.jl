function vortex(x,y,t,γ=1.4)

    x0 = 5
    y0 = 0
    beta = 5
    r2 = @. (x-x0-t)^2 + (y-y0)^2

    u = @. 1 - beta*exp(1.0-r2)*(y-y0)/(2.0*pi)
    v = @. beta*exp(1.0-r2)*(x-x0-t)/(2.0*pi)
    rho = @. 1.0 - (1.0/(8.0*γ*pi^2))*(γ-1.0)/2.0*(beta*exp(1.0-r2))^2
    rho = @. rho^(1/(γ-1))
    p = @. rho^γ

    return (rho, u, v, p)
end
