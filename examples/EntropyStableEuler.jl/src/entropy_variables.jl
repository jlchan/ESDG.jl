unorm(U) = sum(map((x->x.^2),U))
# unorm(U::NTuple{1,T}) where {T} = first(U).^2 # specialize for singleton tuples
# unorm(U::AbstractArray) = U.^2 # scalar case
# unorm(U::AbstractFloat) = U^2 # scalar case

"function primitive_to_conservative_nd(rho,u,v,p)

    convert primitive variables (ρ,U,p) to conservative vars (ρ,ρU,E).
    n-dimensional version where U = tuple(u1,...,u_d)"
function primitive_to_conservative_nd(rho,U,p)
    rhoU = (x->rho.*x).(U)
    Unorm = unorm(U)
    E = @. p/(γ-1) + .5*rho*Unorm
    return (rho,rhoU,E)
end

#####
##### functions of conservative variables
#####

"function pfun_nd(rho, rhoU, E)
    pressure as a function of conservative variables (n-dimensional version).
    n-dimensional version where U = tuple(u1,...,u_d)"
function pfun_nd(rho, rhoU, E)
    rhoUnorm2 = unorm(rhoU)./rho
    return @. (γ-1)*(E - .5*rhoUnorm2)
end

"function betafun_nd(rho,rhoU,E)
    inverse temperature (used in entropy conservative fluxes)"
function betafun_nd(rho,rhoU,E)
    p = pfun_nd(rho,rhoU,E)
    return (@. rho/(2*p))
end

# "function rhoe_ufun_nd(rho, rhoU, E)
#     specific energy as a function of conservative variables"
# function rhoe_ufun_nd(rho, rhoU, E)
#     return pfun_nd(rho, rhoU, E) / (γ-1)
# end

"function sfun(rho, rhoU, E)
    Specific entropy as a function of conservative variables"
function sfun_nd(rho, rhoU, E)
    p = pfun_nd(rho, rhoU, E)
    return @. log(p/(rho^γ))
end

"function Sfun(rho,rhoU,E)
    Mathematical entropy as a function of conservative variables"
function Sfun_nd(rho, rhoU, E)
    return -rho.*sfun_nd(rho, rhoU, E)*entropy_scaling
end

"function v_ufun(rho, rhoU, E)
    Entropy variables as functions of conservative vars"
function v_ufun_nd(rho, rhoU, E)
    s = sfun_nd(rho,rhoU,E)
    p = pfun_nd(rho,rhoU,E)

    v1 = (@. (γ + 1 - s) - (γ-1)*E/p)
    vU = (x->@. x*(γ-1)/p).(rhoU)
    vE = (x->@. x*(γ-1)/p)(-rho)

    # v1,vU,vE = scale_entropy_output(v1, vU, vE)
    return v1,vU,vE
end

#####
##### functions of entropy variables
#####
"function s_vfun(v1,vU,vE)
    entropy as function of entropy variables"
function s_vfun_nd(v1,vU,vE)
    vUnorm = unorm(vU)
    return @. γ - v1 + vUnorm/(2*vE)
end

"function rhoe_vfun(v1,vU,vE)
    specific energy as function of entropy variables"
function rhoe_vfun_nd(v1,vU,vE)
    s = s_vfun_nd(v1,vU,vE)
    return (@. ((γ-1)/((-vE)^γ))^(1/(γ-1)) * exp(-s/(γ-1)))
end

"function u_vfun(v1,vU,vE)
    Conservative vars as functions of entropy variables"
function u_vfun_nd(v1,vU,vE)
    # v1,vU,vE = scale_entropy_input(v1,vU,vE)
    rhoeV     = rhoe_vfun_nd(v1,vU,vE)
    vUnorm    = unorm(vU)
    rho       = (@. rhoeV*(-vE))
    rhoU      = (x->rhoeV.*x).(vU)
    E         = (@. rhoeV*(1-vUnorm/(2*vE)))
    return (rho,rhoU,E)
end

"function conservative_to_primitive_beta_nd(rho,rhoU,E)
    converts conservative variables to `primitive' variables which make
    evaluating EC fluxes simpler."
function conservative_to_primitive_beta_nd(rho,rhoU,E)
    return rho, (x->x./rho).(rhoU), betafun_nd(rho,rhoU,E)
end


# function dUdV_explicit(v1,vU1,vU2,vE)
#     rho,rhou,rhov,E = u_vfun(v1,vU1,vU2,vE)
#     u,v = (x->x./rho).((rhou,rhov))
#     p = pfun(rho,rhou,rhov,E)
#     a2 = γ*p/rho
#     H = a2/(γ-1) + (u^2+v^2)/2
#
#     dUdV = @SMatrix [rho  rhou        rhov        E;
#                      rhou rhou*u + p  rhou*v      rhou*H;
#                      rhov rhov*u      rhov*v + p  rhov*H;
#                      E    rhou*H      rhov*H      rho*H^2-a2*p/(γ-1)]
#
#     return dUdV*(1/(γ-1))
# end
#
# function dVdU_explicit(rho,rhou,rhov,E)
#     rhoe = rhoe_ufun(rho,rhou,rhov,E)
#     V = v_ufun(rho,rhou,rhov,E)
#     k = .5*(V[2]^2+V[3]^2)/V[4]
#
#     dVdU = @SMatrix [γ+k^2      k*V[2]          k*V[3]         V[4]*(k+1);
#                     k*V[2]      V[2]^2-V[4]     V[2]*V[3]      V[2]*V[4];
#                     k*V[3]      V[2]*V[3]       V[3]^2-V[4]    V[3]*V[4]
#                     V[4]*(k+1)  V[2]*V[4]       V[3]*V[4]      V[4]^2]
#     return -dVdU/(rhoe*V[4])
# end
