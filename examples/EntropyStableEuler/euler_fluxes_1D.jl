#####
##### helper functions
#####

"function wavespeed_1D(rho,rhou,E)
    one-dimensional wavespeed (for DG penalization terms)"
function wavespeed_1D(rho,rhou,E)
    p = pfun_nd(rho,(rhou,),E)
    cvel = @. sqrt(γ*p/rho)
    return @. abs(rhou/rho) + cvel
end

#####
##### one-dimensional fluxes
#####

"function euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
                        rhoR,uR,betaR,rhologR,betalogR)"
function euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
                        rhoR,uR,betaR,rhologR,betalogR)

    rholog = logmean.(rhoL,rhoR,rhologL,rhologR)
    betalog = logmean.(betaL,betaR,betalogL,betalogR)

    # arithmetic avgs
    rhoavg = (@. .5*(rhoL+rhoR))
    uavg   = (@. .5*(uL+uR))

    unorm = (@. uL*uR)
    pa    = (@. rhoavg/(betaL+betaR))
    f4aux = (@. rholog/(2*(γ-1)*betalog) + pa + .5*rholog*unorm)

    FxS1 = (@. rholog*uavg)
    FxS2 = (@. FxS1*uavg + pa)
    FxS3 = (@. f4aux*uavg)

    return (FxS1,FxS2,FxS3)
end


#####
##### shim functions for 1D
#####

# dispatch to n-dimensional constitutive routines, with optional entropy scaling
function primitive_to_conservative(rho,u,p)
   rho,rhou,E = primitive_to_conservative_nd(rho,tuple(u),p)
   return rho,first(rhou),E
end
function v_ufun(rho,rhou,E)
    v1,vU,vE = v_ufun_nd(rho,tuple(rhou),E)
    return scale_entropy_output(v1,first(vU),vE)
end
function u_vfun(v1,vU,vE)
    v1,vU,vE = scale_entropy_input(v1,vU,vE)
    rho,rhoU,E = u_vfun_nd(v1,tuple(vU),vE)
    return rho,first(rhoU),E
end
function conservative_to_primitive_beta(rho,rhou,E)
    rho,U,beta = conservative_to_primitive_beta_nd(rho,tuple(rhou),E)
    return rho,first(U),beta
end

Sfun(rho,rhou,E) = Sfun_nd(rho,tuple(rhou),E)
pfun(rho,rhou,E) = pfun_nd(rho,tuple(rhou),E)
betafun(rho,rhou,E) = betafun_nd(rho,tuple(rhou),E)

"function euler_fluxes_1D(rhoL,uL,betaL,rhoR,uR,betaR)"
function euler_fluxes(rhoL,uL,betaL,rhoR,uR,betaR)
    rhologL,betalogL,rhologR,betalogR = map(x->log.(x),(rhoL,betaL,rhoR,betaR))
    return euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,
                           rhoR,uR,betaR,rhologR,betalogR)
end
euler_fluxes(rhoL,uL,betaL,rhologL,betalogL,rhoR,uR,betaR,rhologR,betalogR) =
    euler_fluxes_1D(rhoL,uL,betaL,rhologL,betalogL,rhoR,uR,betaR,rhologR,betalogR)
