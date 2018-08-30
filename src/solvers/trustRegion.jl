export trustRegion
@doc doc"""
    trustRegion(M,F,∇F,[x])
perform the trust region algorithm on the [`Manifold`](@ref)` M` for the
funtion ` F`
"""
function trustRegion(M::mT,
        F::Function, ∇F::Function, x::MP=randomPoint(M);
        TrustRegionSubSolver::Function=truncatedConjugateGradient,
        maxTrustRadius = typicalDistance(M),
        initialTrustRadius = maxTrustRadius / 8,
        retraction::Function = exp,
        stoppingCriterion::Function = (i,ξ,x,xnew) -> (norm(M,x,ξ) < 10.0^-4 || i > 499, (i>499) ? "max Iter $(i) reached." : "critical point reached"),
        subStoppingCriterion::Function =
        returnReason=false,
        minΡAccept=0.1,
        kwargs... #especially may contain debug
    ) where {mT <: Manifold, MP <: MPoint}
    p = GradientProblem(M,F,∇F)
    o = TrustRegionOptions(x,initTrustRadius, maxTrustRadius,minΡAccept,stoppingCriterion,retraction,TrustRegionSubSolver)
    # create default here to check if the user provided a debug and still have the typecheck
    debug::Tuple{Function,Dict{String,Any},Int}= (x::Dict{String,Any}->print(""),Dict{String,Any}(),0);
    kwargs=Dict(kwargs)
    if haskey(kwargs, :debug) # if a key is given -> decorate Options.
        debug = kwargs[:debug]
        o = DebugDecoOptions(o,debug[1],debug[2],debug[3])
    end
    x,r = TrustRegion(p,o)
    if returnReason
        return x,r;
    else
        return x;
    end
end
"""
    trustRegion(p,o)
perform a trust region algorithm based on a [`GradientProblem`](@ref) or a [`HessianProblem`](@ref)
together with some [`TrustRegionOptions`](@ref)
"""
function trustRegion(p::Pr,x::P,o::O) where {Pr <: Union{GradientProblem, HessianProblem}, P <: MPoint, O <: Options}
  M = p.M;
  retr = getOptions(o).retraction
  Δ = getTrustRadius(o);
  Δmax = getOptions(o).maxTrustRadius;
  ρAccept = getOptions(o).minΡAccept;
  tRSub = getOptions(o).TrustRegionSubSolver;
  tRSubO = getOptions(o).TrustRegionSubOptions;
  stop = false;
  iter=0;
  while !stop
    x = xnew;
    iter = iter+1;
    η = tRSub(p,x,tRSubO)
    Hη = HessF(p,x,η)
    xTest = retr(M,x,η)
    ρ = (getCost(p,x) - getCost(p,xTest)) / ( dot(M,x, getGradient(p,x),η) + 0.5*dot(M,η,Hη) )
    if ρ < 1/4 # bad approximation -> decrease trust region
        Δ = Δ/4;
    elseif (ρ > 3/4) && (norm(M,x,η) == Δ) # good approximation _and_ step at boundary of trust -> increase trust
        Δ = min(2*Δ,Δmax);
    end # otherwise just keep trust
    if ρ > ρAccept
        xnew = xTest
    end
    # update Options
    updateTrustRadius!(tRSubO,Δ);
    stop,reason = evaluateStoppingCriterion(o,iter,η,x,xnew)
    x = xnew
  end
  return x,reason
end

function TrustRegionConjugateGradient(p::Pr,x::P,o::O) where {Pr <: Problem, P <: MPoint, O <: Options}
    ηnew = zeroTVector(x); η = ηnew
    rnew = getGradient(p,x); r = newr; δnew = -r
    m = (η,Hη) -> dot(p.M,x,η,r) + 0.5*dot(p.M,x,η,Hη)
    Δ = getTrustRadius(o)
    stop = false;
    iter = 0;
    while !stop
        η = ηnew # save last η
        iter = iter+1;
        # update eta
        dotR = dot(M,x,r,r)
        α = dotR / dot(M,δ,Hδ)
        ηnew = η + α*δ
        #
        Hδ = HessianF(p,x,δ)
        if dot(M,x,δ,Hδ) ≤ 0  || norm(M,x,η) ≥ Δ
            # Formula fomm Sec. 3 [ABG04]
            τ = 1/dot(M,x,δ,δ)*(-dot(M,x,η,δ) + sqrt(dot(M,x,η,δ)^2 - (Δ^2 - dot(M,x,η,η))*dot(M,x,δ,δ) ) )
            return η + τ*δ
        end
        # Update r and δ
        r = r + α*Hδ
        δ = -r + dot(M,x,r,r)/dotR*δ
        stop, reason = evaluateStoppingCriterion(o,iter,x,η,ηnew)
    end
    return η
end
