#
# Options
#
@doc raw"""
    ALMOptions{P,T} <: Options

Describes the augmented Lagrangian method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a set point on a manifold as starting point

# Constructor

    ALMOptions(x)

construct an augmented Lagrangian Option with the fields and defaults as above.

# See also
[`augmented_Lagrangian_method`](@ref)
"""
mutable struct ALMOptions{P, Pr <: Problem, Op <: Options, TStopping <: StoppingCriterion} <: Options
    x::P
    sub_problem::Pr 
    sub_options::Op 
    max_inner_iter::Int
    num_outer_itertgn::Int
    ϵ::Real #(starting)tolgradnorm
    ϵ_min::Real #endingtolgradnorm
    bound::Real
    #### multiplier boundaries anpassen
    λ::Vector
    γ::Vector
    ρ::Real
    τ::Real
    θ_ρ::Real
    stop::TStopping 
    function ALMOptions(
        M::AbstractManifold,
        p::ConstrainedProblem,
        x0::P,
        sub_problem::Pr, 
        sub_options::Op; 
        max_inner_iter::Int=200,
        num_outer_itertgn::Int=30,
        ϵ::Real=1e-3, #(starting)tolgradnorm
        ϵ_min::Real=1e-6, #endingtolgradnorm
        bound::Real=20.0,
        #### multiplier boundaries anpassen
        λ::Vector=ones(len(get_inequality_constraints(p,x0))),
        γ::Vector=ones(len(get_equality_constraints(p,x0))),
        ρ::Real=1.0, 
        τ::Real=0.8,
        θ_ρ::Real=0.3, 
        stopping_criterion::StoppingCriterion=StopWhenAny(StopAfterIteration(300), StopWhenAll(StopIfSmallerOrEqual(ϵ, ϵ_min), StopWhenChangeLess(1e-6))),
    ) where {P, Pr <: Problem, Op <: Options} 
        o = new{
            P,
            Pr,
            Op,
            typeof(stopping_criterion),
        }()
        o.x = x0
        o.sub_problem = sub_problem
        o.sub_options = sub_options
        o.max_inner_iter = max_inner_iter
        o.num_outer_itertgn = num_outer_itertgn
        o.ϵ = ϵ
        o.ϵ_min = ϵ_min
        o.bound = bound
        #### multiplier boundaries anpassen
        o.λ = λ
        o.γ = γ
        o.ρ = ρ
        o.τ = τ
        o.θ_ρ = θ_ρ
        o.θ_ϵ = 0.0
        o.old_acc = 0.0
        o.stop = stopping_criterion
        return o
    end
end