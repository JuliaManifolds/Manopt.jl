#
# Options
#
@doc raw"""
    EPMOptions{P,T} <: Options

Describes the exact penalty method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a set point on a manifold as starting point
* `smoothing_technique` – a smoothing technique with which the penalized objective can be smoothed
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `max_inner_iter` – (`200`) the maximum number of iterations the subsolver should perform in each iteration 
* `num_outer_itertgn` – (`30`)
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `ρ` – (`1.0`) the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `θ_ϵ` – (`(ϵ_min/ϵ)^(1/num_outer_itertgn)`) the scaling factor of the accuracy tolerance
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(tolgradnorm, ending_tolgradnorm), `[`StopWhenChangeLess`](@ref)`(1e-6)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.


# Constructor

    EPMOptions(x)

construct an exact penalty Option with the fields and defaults as above.

# See also
[`exact_penalty_method`](@ref)
"""
mutable struct EPMOptions{P, Pr <: Problem, Op <: Options, TStopping <: StoppingCriterion} <: Options
    x::P
    smoothing_technique::String
    sub_problem::Pr 
    sub_options::Op 
    max_inner_iter::Int
    num_outer_itertgn::Int
    tolgradnorm::Real 
    ending_tolgradnorm::Real
    ϵ::Real
    ϵ_min::Real 
    ρ::Real
    θ_ρ::Real
    θ_ϵ::Real
    θ_tolgradnorm::Real
    stop::TStopping 
    function EPMOptions(
        M::AbstractManifold,
        p::ConstrainedProblem,
        x0::P,
        smoothing_technique::String,
        sub_problem::Pr, 
        sub_options::Op; 
        max_inner_iter::Int=200,
        num_outer_itertgn::Int=30,
        tolgradnorm::Real=1e-3,
        ending_tolgradnorm::Real=1e-6,
        ϵ::Real=1e-1,
        ϵ_min::Real=1e-6, 
        ρ::Real=1.0, 
        θ_ρ::Real=0.3, 
        stopping_criterion::StoppingCriterion=StopWhenAny(StopAfterIteration(300), StopWhenAll(StopWhenSmallerOrEqual(:tolgradnorm, ending_tolgradnorm), StopWhenChangeLess(1e-6))),
    ) where {P, Pr <: Problem, Op <: Options} 
        o = new{
            P,
            Pr,
            Op,
            typeof(stopping_criterion),
        }()
        o.x = x0
        o.smoothing_technique = smoothing_technique
        o.sub_problem = sub_problem
        o.sub_options = sub_options
        o.max_inner_iter = max_inner_iter
        o.num_outer_itertgn = num_outer_itertgn
        o.tolgradnorm = tolgradnorm
        o.ending_tolgradnorm = ending_tolgradnorm
        o.ϵ = ϵ
        o.ϵ_min = ϵ_min
        o.ρ = ρ
        o.θ_ρ = θ_ρ
        o.θ_ϵ = 0.0
        o.θ_tolgradnorm = 0.0
        o.stop = stopping_criterion
        return o
    end
end