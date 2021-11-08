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
    num_outer_itertgn::Int
    ϵ::Real #(starting)tolgradnorm
    ϵ_min::Real #endingtolgradnorm
    bound::Real
    λ::Vector
    γ::Vector
    ρ::Real
    τ::Real
    θ_ρ::Real
    stop::TStopping
    function ALMOptions(
        x0::P,
        n_ineq::Int,
        n_eq::Int,
        sub_problem::Pr,
        sub_options::Op,
        num_outer_itertgn::Int=30,
        ϵ::Real=1e-3, #(starting)tolgradnorm
        ϵ_min::Real=1e-6, #endingtolgradnorm
        bound::Real=20.0,
        λ::Vector=ones(n_ineq),
        γ::Vector=ones(n_eq),
        ρ::Real=1.0, 
        τ::Real=0.8,
        θ_ρ::Real=0.3, 
        stopping_criterion::StoppingCriterion=StopWhenAny(StopAfterIteration(300), StopWhenAll(StopIfSmallerOrEqual(ϵ, ϵ_min), StopWhenChangeLess(1e-6))),
    ) where {P, Pr <: Problem, Op <: Options} 
        θ_ϵ=(ϵ_min/ϵ)^(1/num_outer_itertgn), 
        o = new{
            P,
            Pr,
            Op,
            typeof(stopping_criterion),
        }()
        o.x = x0
        o.sub_problem,
        o.sub_options,
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
        o.old_acc = old_acc
        o.stop = stopping_criterion
        return o
    end
end