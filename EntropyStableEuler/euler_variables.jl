"""
function ??

"""

"primitive pressure to conservative vars"
function primitive_to_conservative(rho,u,v,p)
    rhou = @. rho*u
    rhov = @. rho*v
    E = @. p/(γ-1) + .5*rho*(u^2+v^2)
    return (rho,rhou,rhov,E)
end

"inverse temperature (used in entropy conservative fluxes)"
function betafun(rho,u,v,E,γ=1.4)
    return rho/(2*pfun(rho,u,v,E,γ))
end

"pressure as a function of ρ,u,v,E"
function pfun(rho,u,v,E,γ=1.4)
    return (γ-1)*(E-.5*rho*(u^2+v^2))
end

"specific energy as a function of conservative variables"
function rhoefun(rho,rhou,rhov,E)
    return E - .5*(rhou^2+rhov^2)/rho
end

"Thermodynamic entropy as a function of conservative variables"
function sfun(rho,rhou,rhov,E,γ=1.4)
    return log((γ-1)*rhoefun(rho,rhou,rhov,E)/(rho^γ))
end

"Entropy variables as functions of conservative vars"
function v_ufun(rho,rhou,rhov,E,γ=1.4)
    ρe = rhoefun.(rho,rhou,rhov,E)
    v1 = @. (-E + ρe*(γ + 1 - sfun(rho,rhou,rhov,E)))/ρe
    v2 = @. rhou/ρe
    v3 = @. rhov/ρe
    v4 = @. (-rho)/ρe
    return v1,v2,v3,v4
end

"entropy as function of entropy variables"
function s_vfun(v1,v2,v3,v4,γ=1.4)
    return γ - v1 + (v2^2+v3^2)/(2*v4)
end

"specific energy as function of entropy variables"
function rhoe_vfun(v1,v2,v3,v4,γ=1.4)
    return ((γ-1)/((-v4)^γ))^(1/(γ-1)) * exp(-s_vfun(v1,v2,v3,v4)/(γ-1))
end

"Conservative vars as functions of entropy variables"
function u_vfun(v1,v2,v3,v4,γ=1.4)
    rhoeV = rhoe_vfun.(v1,v2,v3,v4)
    rho   = @. rhoeV.*(-v4)
    rhou  = @. rhoeV*(v2)
    rhov  = @. rhoeV*(v3)
    E     = @. rhoeV*(1-(v2^2+v3^2)/(2.0*v4))
    return rho,rhou,rhov,E
end
