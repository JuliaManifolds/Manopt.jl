#
# Options
#
@doc raw"""
    RALMOptions{P,T} <: Options

Describes the Riemannian augmented Lagrangian method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a set point on a manifold as starting point

# Constructor

    RALMOptions(x)

construct a Riemannian augmented Lagrangian Option with the fields and defaults as above.

# See also
[`Riemannian_augmented_Lagrangian_method`](@ref)
"""
mutable struct RALMOptions{
    x::P,
    
} <: Options
    x::P

    stop::TStopping
    
    function RALMOptions(
        x0::random_point(M),
        max_inner_iter=200,
        num_outer_itertgn=30,
        ϵ=1e-3, #(starting)tolgradnorm
        ϵ_min=1e-6, #endingtolgradnorm
        bound=20, ###why not Real?
        λ::Vector=ones(n_ineq_constraint,1),
        γ::Vector=ones(n_eq_constraint,1),
        ρ=1.0, ###why int in Matlab code?
        τ=0.8,
        θ_ρ=0.3, 
        θ_ϵ=(ϵ_min/ϵ)^(1/num_outer_itertgn), ###this does not need to be a parameter, just defined somewhere
        oldacc=Inf, ###this does not need to be a parameter, just defined somewhere
        stopping_criterion::StoppingCriterion=StopAfterIteration(300), #maxOuterIter
    )
        o = new{
            typeof(x0),
            typeof(max_inner_iter + num_outer_itertgn),
            typeof(ϵ + ϵ_min + ρ + τ + θ_ρ + θ_ϵ + oldacc),
            typeof(bound),
            typeof(λ),
            typeof(γ),
            typeof(stopping_criterion),
        }()
        o.x = x0
        o.max_inner_iter = max_inner_iter
        o.num_outer_itertgn = num_outer_itertgn
        o.ϵ = ϵ
        o.ϵ_min = ϵ_min
        o.bound = bound
        o.λ = λ
        o.γ = γ
        o.ρ = ρ
        o.τ = τ
        o.θ_ρ = θ_ρ
        o.θ_ϵ = θ_ϵ
        o.oldacc = oldacc
        o.stop = stopping_criterion
        return o
    end
end