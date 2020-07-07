"""
Euler functions for entropy variables and more
"""

"wavespeed(rho,rhou,E): 1D wavespeed for use in interface fluxes"
function wavespeed(rho,rhou,E)
    cvel = (@. sqrt(γ*pfun(rho,rhou,E)/rho))
    return (@. abs(rhou/rho) + cvel)
end

"vector_norm(U): computes norm of vector-valued variables
Example: vector_norm((randn(10),randn(10)))"
vector_norm(U) = sum((x->x.^2).(U))


"primitive pressure to conservative vars"
function primitive_to_conservative(rho,u,v,p)
    U = (u,v)
    unorm = vector_norm(U)
    rhou,rhov = (x->rho.*x).(U)
    E = (@. p/(γ-1) + .5*rho*unorm)
    return (rho,rhou,rhov,E)
end

"inverse temperature (used in entropy conservative fluxes)"
function betafun(rho,rhou,rhov,E)
    p = pfun(rho,rhou,rhov,E)
    return (@. rho/(2*p))
end

"pressure as a function of ρ,u,v,E"
function pfun(rho,rhou,rhov,E)
    # rhoU = (rhou,rhov)
    rhounorm = @. (rhou^2+rhov^2)/rho
    return @. (γ-1)*(E-.5*rhounorm)
end

"1D pressure function"
function pfun(rho,rhou,E)
    rhounorm = @. rhou^2/rho
    return @. (γ-1)*(E-.5*rhounorm)
end

"specific energy as a function of conservative variables"
function rhoe_ufun(rho,rhou,rhov,E)
    rhoU = (rhou,rhov)
    rhoUnorm = vector_norm(rhoU)
    return (@. E - .5*rhoUnorm/rho)
end

# Thermodynamic entropy as a function of conservative variables"
function sfun(rho,rhou,rhov,E)
    rhoe = rhoe_ufun(rho,rhou,rhov,E)
    return (@. log((γ-1)*rhoe/(rho^γ)))
    # return (@. log(abs((γ-1)*rhoe/(rho^γ))))
end

# Mathematical entropy
function Sfun(rho,rhou,E)
    return -rho.*sfun(rho,(rhou,zeros(size(rhou))),E)
end

# Entropy variables as functions of conservative vars
function v_ufun(rho,rhou,rhov,E)
    ρe = rhoe_ufun(rho,rhou,rhov,E)
    sU = sfun(rho,rhou,rhov,E)
    v1 = (@. (-E + ρe*(γ + 1 - sU))/ρe)
    vU1,vU2 = (x->x./ρe).((rhou,rhov))
    vE = (@. (-rho)/ρe)
    return (v1,vU1,vU2,vE)
end

# entropy as function of entropy variables"
function s_vfun(v1,vU1,vU2,vE)
    vUnorm = vector_norm((vU1,vU2))
    return (@. γ - v1 + vUnorm/(2*vE))
end

# specific energy as function of entropy variables"
function rhoe_vfun(v1,vU1,vU2,vE)
    s = s_vfun(v1,vU1,vU2,vE)
    return (@. ((γ-1)/((-vE)^γ))^(1/(γ-1)) * exp(-s/(γ-1)))
end

# Conservative vars as functions of entropy variables"
function u_vfun(v1,vU1,vU2,vE)
    vU = (vU1,vU2)
    rhoeV = rhoe_vfun(v1,vU1,vU2,vE)
    vUnorm = vector_norm(vU)
    rho   = (@. rhoeV*(-vE))
    rhou,rhov = (x->rhoeV.*x).(vU)
    E     = (@. rhoeV*(1-vUnorm/(2*vE)))
    return (rho,rhou,rhov,E)
end

function Sfun(rho,rhou,rhov,E)
    return -rho.*sfun(rho,rhou,rhov,E)
end

function dUdV_explicit(v1,vU1,vU2,vE)

    rho,rhou,rhov,E = u_vfun(v1,vU1,vU2,vE)
    u,v = (x->x./rho).((rhou,rhov))
    p = pfun(rho,rhou,rhov,E)
    a2 = γ*p/rho
    H = a2/(γ-1) + (u^2+v^2)/2

    dUdV = @SMatrix [rho  rhou       rhov        E;
                    rhou rhou*u + p rhou*v      rhou*H;
                    rhov rhov*u     rhov*v + p  rhov*H;
                    E    rhou*H     rhov*H      rho*H^2-a2*p/(γ-1)]

    return dUdV*(1/(γ-1))
end

function dVdU_explicit(rho,rhou,rhov,E)
    #rhoev = rhoe_vfun(V[1],(V[2],V[3]),V[4])
    rhoe = rhoe_ufun(rho,rhou,rhov,E)
    V = v_ufun(rho,rhou,rhov,E)
    k = .5*(V[2]^2+V[3]^2)/V[4]

    dVdU = @SMatrix [γ+k^2     k*V[2]          k*V[3]         V[4]*(k+1);
                    k*V[2]     V[2]^2-V[4]     V[2]*V[3]     V[2]*V[4];
                    k*V[3]     V[2]*V[3]      V[3]^2-V[4]    V[3]*V[4]
                    V[4]*(k+1) V[2]*V[4]       V[3]*V[4]      V[4]^2]
    return -dVdU/(rhoe*V[4])
end


# # 1D wavespeed for use in interface fluxes
# function wavespeed(rho,rhou,E)
#     cvel = (@. sqrt(γ*pfun(rho,rhou,E)/rho))
#     return (@. sqrt(abs(rhou/rho)) + cvel)
# end
#
# vector_norm(U) = sum((x->x.^2).(U))
#
# "primitive pressure to conservative vars"
# function primitive_to_conservative(rho,U,p)
#     rhoU = (x->rho.*x).(U)
#     unorm = vector_norm(U)
#     E = (@. p/(γ-1) + .5*rho*unorm)
#     return (rho,rhoU...,E)
# end
#
# "inverse temperature (used in entropy conservative fluxes)"
# function betafun(rho,rhoU,E)
#     p = pfun(rho,rhoU,E)
#     return (@. rho/(2*p))
# end
#
# "pressure as a function of ρ,u,v,E"
# function pfun(rho,rhoU,E)
#     rhounorm = vector_norm(rhoU)./rho
#     return @. (γ-1)*(E-.5*rhounorm)
# end
# function pfun(rho,rhou,rhov,E)
#     return pfun(rho,(rhou,rhov),E)
# end
#
# # # hack
# # function pressure_fun(rho,rhou,rhov,E)
# #     rhounorm = @. (rhou^2+rhov^2)/rho
# #     return @. (γ-1)*(E-.5*rhounorm)
# # end
#
# # specific energy as a function of conservative variables
# function rhoe_ufun(rho,rhoU,E)
#     rhoUnorm = vector_norm(rhoU)
#     return (@. E - .5*rhoUnorm/rho)
# end
#
# # Thermodynamic entropy as a function of conservative variables"
# function sfun(rho,rhoU,E)
#     rhoe = rhoe_ufun(rho,rhoU,E)
#     return (@. log((γ-1)*rhoe/(rho^γ)))
# end
#
# # Mathematical entropy
# function Sfun(rho,rhou,E)
#     return -rho.*sfun(rho,(rhou,zeros(size(rhou))),E)
# end
#
# # Entropy variables as functions of conservative vars
# function v_ufun(rho,rhoU,E)
#     ρe = rhoe_ufun(rho,rhoU,E)
#     sU = sfun(rho,rhoU,E)
#     v1 = (@. (-E + ρe*(γ + 1 - sU))/ρe)
#     #vU = [@. rhoUi/ρe for rhoUi in rhoU]
#     vU = (x->x./ρe).(rhoU)
#     vE = (@. (-rho)/ρe)
#     return (v1,vU...,vE)
# end
#
# # entropy as function of entropy variables"
# function s_vfun(v1,vU,vE)
#     vUnorm = vector_norm(vU)
#     return (@. γ - v1 + vUnorm/(2*vE))
# end
#
# # specific energy as function of entropy variables"
# function rhoe_vfun(v1,vU,vE)
#     s = s_vfun(v1,vU,vE)
#     return (@. ((γ-1)/((-vE)^γ))^(1/(γ-1)) * exp(-s/(γ-1)))
# end
#
# # Conservative vars as functions of entropy variables"
# function u_vfun(v1,vU,vE)
#     rhoeV = rhoe_vfun(v1,vU,vE)
#     vUnorm = vector_norm(vU)
#     rho   = (@. rhoeV*(-vE))
#     #rhoU  = [@. rhoeV*(vUi) for vUi in vU]
#     rhoU = (x->rhoeV.*x).(vU)
#     E     = (@. rhoeV*(1-vUnorm/(2*vE)))
#     return (rho,rhoU...,E)
# end
#
#
# # specialize in 1D
# function v_ufun1D(rho,rhou,E)
#     v1,v2,v3,v4 = v_ufun(rho,rhou,zeros(size(rhou)),E)
#     return v1,v2,v4
# end
#
#
# function primitive_to_conservative(rho,u,v,p)
#     return primitive_to_conservative(rho,(u,v),p)
# end
# function primitive_to_conservative(rho,u,v,w,p)
#     return primitive_to_conservative(rho,(u,v,w),p)
# end
# function betafun1D(rho,rhou,E)
#     return betafun(rho,rhou,zeros(size(rhou)),E)
# end
# function betafun(rho,rhou,rhov,E)
#     return betafun(rho,(rhou,rhov),E)
# end
# # function betafun(rho,rhou,rhov,rhow,E)
# #     return betafun(rho,(rhou,rhov,rhow),E)
# # end
# function Sfun(rho,rhou,rhov,E)
#     return -rho.*sfun(rho,(rhou,rhov),E)
# end
# function Sfun(rho,rhou,rhov,rhow,E)
#     return -rho.*sfun(rho,(rhou,rhov,rhow),E)
# end
# function v_ufun(rho,rhou,rhov,E)
#     return v_ufun(rho,(rhou,rhov),E)
# end
# function v_ufun(rho,rhou,rhov,rhow,E)
#     return v_ufun(rho,(rhou,rhov,rhow),E)
# end
# # more expensive but since 1D should be OK
# function u_vfun1D(v1,v2,v3)
#     u1,u2,u3,u4 = u_vfun(v1,v2,zeros(size(v2)),v3)
#     return u1,u2,u4
# end
# function u_vfun(v1,v2,v3,v4)
#     return u_vfun(v1,(v2,v3),v4)
# end
# function u_vfun(v1,v2,v3,v4,v5)
#     return u_vfun(v1,(v2,v3,v4),v5)
# end
#
# function dUdV_explicit(V)
#
#     rho,rhou,rhov,E = u_vfun(V[1],V[2],V[3],V[4])
#     u,v = (x->x./rho).((rhou,rhov))
#     p = pfun(rho,rhou,rhov,E)
#     a2 = γ*p/rho
#     H = a2/(γ-1) + (u^2+v^2)/2
#
#     dUdV = @SMatrix [rho  rhou       rhov        E;
#                     rhou rhou*u + p rhou*v      rhou*H;
#                     rhov rhov*u     rhov*v + p  rhov*H;
#                     E    rhou*H     rhov*H      rho*H^2-a2*p/(γ-1)]
#
#     return dUdV*(1/(γ-1))
# end
#
# function dVdU_explicit(U)
#     #rhoev = rhoe_vfun(V[1],(V[2],V[3]),V[4])
#     rhoe = rhoe_ufun(U[1],(U[2],U[3]),U[4])
#     V = v_ufun(U[1],(U[2],U[3]),U[4])
#     k = .5*(V[2]^2+V[3]^2)/V[4]
#
#     dVdU = @SMatrix [γ+k^2     k*V[2]          k*V[3]         V[4]*(k+1);
#                     k*V[2]     V[2]^2-V[4]     V[2]*V[3]     V[2]*V[4];
#                     k*V[3]     V[2]*V[3]      V[3]^2-V[4]    V[3]*V[4]
#                     V[4]*(k+1) V[2]*V[4]       V[3]*V[4]      V[4]^2]
#     return -dVdU/(rhoe*V[4])
# end

# function dUdV_explicit(V)
#     dUdV = zeros(4,4)
#     rho,rhou,rhov,E = u_vfun(V...)
#     u,v = (x->x./rho).((rhou,rhov))
#     p = pfun(rho,rhou,rhov,E)
#     a2 = γ*p/rho
#     H = a2/(γ-1) + (u^2+v^2)/2
#
#     dUdV[1,1] = rho
#     dUdV[1,2] = rhou
#     dUdV[1,3] = rhov
#     dUdV[1,4] = E
#
#     dUdV[2,1] = rhou
#     dUdV[2,2] = rhou*u + p
#     dUdV[2,3] = rhou*v
#     dUdV[2,4] = rhou*H
#
#     dUdV[3,1] = rhov
#     dUdV[3,2] = rhov*u
#     dUdV[3,3] = rhov*v + p
#     dUdV[3,4] = rhov*H
#
#     dUdV[4,1] = E
#     dUdV[4,2] = rhou*H
#     dUdV[4,3] = rhov*H
#     dUdV[4,4] = rho*H^2 - a2*p/(γ-1)
#
#     return dUdV*(1/(γ-1))
# end
