"""
Euler functions for entropy variables and more

"""

function Unorm(U)
    unorm = zeros(size(U[1]))
    for u in U
        @. unorm += u^2
    end
    return unorm
end

"primitive pressure to conservative vars"
function primitive_to_conservative(rho,U,p)
    rhoU = [rho.*u for u in U]
    unorm = Unorm(U)
    E = @. p/(γ-1) + .5*rho*unorm
    return (rho,rhoU...,E)
end

"inverse temperature (used in entropy conservative fluxes)"
function betafun(rho,rhoU,E)
    p = pfun(rho,rhoU,E)
    return @. rho/(2*p)
end

"pressure as a function of ρ,u,v,E"
function pfun(rho,rhoU,E,rhounorm)
    return @. (γ-1)*(E-.5*rhounorm)
end
function pfun(rho,rhoU,E)
    rhounorm = Unorm(rhoU)./rho
    return pfun(rho,rhoU,E,rhounorm)
end

"specific energy as a function of conservative variables"
function rhoefun(rho,rhoU,E)
    rhoUnorm = Unorm(rhoU)
    return @. E - .5*rhoUnorm/rho
end

"Thermodynamic entropy as a function of conservative variables"
function sfun(rho,rhoU,E)
    rhoe = rhoefun(rho,rhoU,E)
    return @. log((γ-1)*rhoe/(rho^γ))
end

"Entropy variables as functions of conservative vars"
function v_ufun(rho,rhoU,E)
    ρe = rhoefun(rho,rhoU,E)
    sU = sfun(rho,rhoU,E)
    v1 = @. (-E + ρe*(γ + 1 - sU))/ρe
    vU = [@. rhoUi/ρe for rhoUi in rhoU]
    vE = @. (-rho)/ρe
    return (v1,vU...,vE)
end

"entropy as function of entropy variables"
function s_vfun(v1,vU,vE)
    vUnorm = Unorm(vU)
    return @. γ - v1 + vUnorm/(2*vE)
end

"specific energy as function of entropy variables"
function rhoe_vfun(v1,vU,vE)
    s = s_vfun(v1,vU,vE)
    return @. ((γ-1)/((-vE)^γ))^(1/(γ-1)) * exp(-s/(γ-1))
end

"Conservative vars as functions of entropy variables"
function u_vfun(v1,vU,vE)
    rhoeV = rhoe_vfun(v1,vU,vE)
    vUnorm = Unorm(vU)
    rho   = @. rhoeV*(-vE)
    rhoU  = [@. rhoeV*(vUi) for vUi in vU]
    E     = @. rhoeV*(1-vUnorm/(2*vE))
    return (rho,rhoU...,E)
end

"shim functions to convert to 2D vs nD"
function v_ufun(rho,rhou,rhov,E)
    return v_ufun(rho,(rhou,rhov),E)
end
function v_ufun(rho,rhou,rhov,rhow,E)
    return v_ufun(rho,(rhou,rhov,rhow),E)
end

function u_vfun(v1,v2,v3,v4)
    return u_vfun(v1,(v2,v3),v4)
end
function u_vfun(v1,v2,v3,v4,v5)
    return u_vfun(v1,(v2,v3,v4),v5)
end

function betafun(rho,rhou,rhov,E)
    return betafun(rho,(rhou,rhov),E)
end
function betafun(rho,rhou,rhov,rhow,E)
    return betafun(rho,(rhou,rhov,rhow),E)
end

function primitive_to_conservative(rho,u,v,p)
    return primitive_to_conservative(rho,(u,v),p)
end
function primitive_to_conservative(rho,u,v,w,p)
    return primitive_to_conservative(rho,(u,v,w),p)
end
