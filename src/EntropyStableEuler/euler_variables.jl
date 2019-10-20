"""
function ??

"""

"inverse temperature (used in entropy conservative fluxes)"
function betafun(rho,u,v,E,γ=1.4)
    return rho/(2*pfun(rho,u,v,E,γ))
end

"pressure as a function of ρ,u,v,E"
function pfun(rho,u,v,E,γ=1.4)
    return (γ-1)*(E-.5*rho*(u^2+v^2))
end

"specific energy as a function of conservative variables"
function rhoefun(U)
    (rho,rhou,rhov,E) = U
    return E - .5*(rhou^2+rhov^2)/rho
end

"Thermodynamic entropy as a function of conservative variables"
function sfun(U,γ=1.4)
    (rho,rhou,rhov,E) = U
    return log((γ-1)*rhoefun(U)/(rho^γ))
end

"Entropy variables as functions of conservative vars"
function VU(U,γ=1.4)
    (rho,rhou,rhov,E) = U
    ρe = rhoefun(U)
    v1 = (-E + ρe*(γ + 1 - sfun(rho,rhou,rhov,E)))/ρe
    v2 = rhou/ρe
    v3 = rhov/ρe
    v4 = (-rho)/ρe
    return v1,v2,v3,v4
end

"entropy as function of entropy variables"
function sVfun(V,γ=1.4)
    (v1,v2,v3,v4)=V
    return γ - v1 + (v2^2+v3^2)/(2*v4)
end

"specific energy as function of entropy variables"
function rhoeVfun(V,γ=1.4)
    (v1,v2,v3,v4)=V
    return ((γ-1)/((-v4)^γ))^(1/(γ-1)) * exp(-sVfun(V)/(γ-1))
end

"Conservative vars as functions of entropy variables"
function UV(V,γ=1.4)
    (v1,v2,v3,v4) = V
    rhoeV = rhoeVfun(V)
    rho  = rhoeV.*(-v4)
    rhou = rhoeV*(v2)
    rhov = rhoeV*(v3)
    E    = rhoeV*(1-(v2^2+v3^2)/(2*v4))
    return rho,rhou,rhov,E
end
