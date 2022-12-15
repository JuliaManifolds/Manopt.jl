"""
    BundleMethodOptions <: Options
stories option values for a [`bundle_method`](@ref) solver

# Fields

* `p` - current iterate
* `stop` – a [`StoppingCriterion`](@ref)
"""
mutable struct BundleMethodOptions{P,TSC<:StoppingCriterion} <: Options
    p::P
    stop::TSC
    function BundleMethodOptions(
        M::TM, p::P; stopping_criterion::SC=StopAfterIteration(5000)
    ) where {TM<:AbstractManifold,P,SC<:StoppingCriterion}
        return new{P,SC}(p, stopping_criterion)
    end
end
