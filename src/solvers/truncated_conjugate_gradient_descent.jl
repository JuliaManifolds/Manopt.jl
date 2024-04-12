@doc raw"""
    TruncatedConjugateGradientState <: AbstractHessianSolverState

describe the Steihaug-Toint truncated conjugate-gradient method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `Y`:                   (`zero_vector(M,p)`) Current iterate, whose type is also used for the other, internal, tangent vector fields
* `stop`:                a [`StoppingCriterion`](@ref).
* `X`:                   the gradient ``\operatorname{grad}f(p)```
* `δ`:                   the conjugate gradient search direction
* `θ`:                   (`1.0`) 1+θ is the superlinear convergence target rate.
* `κ`:                   (`0.1`) the linear convergence target rate.
* `trust_region_radius`: (`injectivity_radius(M)/4`) the trust-region radius
* `residual`:            the gradient of the model ``m(Y)``
* `randomize`:           (`false`)
* `project!`:            (`copyto!`) for numerical stability it is possible to project onto
  the tangent space after every iteration. By default this only copies instead.

# Internal fields

* `Hδ`, `HY`:                 temporary results of the Hessian applied to `δ` and `Y`, respectively.
* `δHδ`, `YPδ`, `δPδ`, `YPδ`: temporary inner products with `Hδ` and preconditioned inner products.
* `z`:                        the preconditioned residual
* `z_r`:                      inner product of the residual and `z`

# Constructor

    TruncatedConjugateGradientState(TpM::TangentSpace, Y=rand(TpM); kwargs...)

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct TruncatedConjugateGradientState{T,R<:Real,SC<:StoppingCriterion,Proj} <:
               AbstractHessianSolverState
    stop::SC
    X::T
    Y::T
    HY::T
    δ::T
    Hδ::T
    δHδ::R
    YPδ::R
    δPδ::R
    YPY::R
    z::T
    z_r::R
    residual::T
    trust_region_radius::R
    model_value::R
    randomize::Bool
    project!::Proj
    initialResidualNorm::Float64
    function TruncatedConjugateGradientState(
        TpM::TangentSpace,
        Y::T=rand(TpM);
        trust_region_radius::R=injectivity_radius(base_manifold(TpM)) / 4.0,
        randomize::Bool=false,
        project!::F=copyto!,
        θ::Float64=1.0,
        κ::Float64=0.1,
        stopping_criterion::StoppingCriterion=StopAfterIteration(
                                                  manifold_dimension(base_manifold(TpM))
                                              ) |
                                              StopWhenResidualIsReducedByFactorOrPower(;
                                                  κ=κ, θ=θ
                                              ) |
                                              StopWhenTrustRegionIsExceeded() |
                                              StopWhenCurvatureIsNegative() |
                                              StopWhenModelIncreased(),
        kwargs...,
    ) where {T,R<:Real,F}
        tcgs = new{T,R,typeof(stopping_criterion),F}()
        tcgs.stop = stopping_criterion
        tcgs.Y = Y
        tcgs.trust_region_radius = trust_region_radius
        tcgs.randomize = randomize
        tcgs.project! = project!
        tcgs.model_value = zero(trust_region_radius)
        return tcgs
    end
end
function show(io::IO, tcgs::TruncatedConjugateGradientState)
    i = get_count(tcgs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(tcgs.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Truncated Conjugate Gradient Descent
    $Iter
    ## Parameters
    * randomize: $(tcgs.randomize)
    * trust region radius: $(tcgs.trust_region_radius)

    ## Stopping criterion

    $(status_summary(tcgs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
function set_manopt_parameter!(tcgs::TruncatedConjugateGradientState, ::Val{:Iterate}, Y)
    return tcgs.Y = Y
end
get_iterate(tcgs::TruncatedConjugateGradientState) = tcgs.Y
function set_manopt_parameter!(
    tcgs::TruncatedConjugateGradientState, ::Val{:TrustRegionRadius}, r
)
    return tcgs.trust_region_radius = r
end

function get_manopt_parameter(
    tcgs::TruncatedConjugateGradientState, ::Val{:TrustRegionExceeded}
)
    return (tcgs.YPY >= tcgs.trust_region_radius^2)
end

#
# Special stopping Criteria
#
@doc raw"""
    StopWhenResidualIsReducedByFactorOrPower <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
either by a power of 1+θ or by a factor κ compared to the norm of the initial
residual. The criterion hence reads
``\Vert r_k \Vert_p \leqq \Vert r_0 \Vert_p \min \bigl( \kappa, \Vert r_0 \Vert_p^θ \bigr)``.

# Fields

* `κ`:      the reduction factor
* `θ`:      part of the reduction power
* `reason`: stores a reason of stopping if the stopping criterion has one be reached,
  see [`get_reason`](@ref).

# Constructor

    StopWhenResidualIsReducedByFactorOrPower(; κ=0.1, θ=1.0)

Initialize the StopWhenResidualIsReducedByFactorOrPower functor to indicate to stop after
the norm of the current residual is lesser than either the norm of the initial residual
to the power of 1+θ or the norm of the initial residual times κ.

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenResidualIsReducedByFactorOrPower <: StoppingCriterion
    κ::Float64
    θ::Float64
    reason::String
    at_iteration::Int
    function StopWhenResidualIsReducedByFactorOrPower(; κ::Float64=0.1, θ::Float64=1.0)
        return new(κ, θ, "", 0)
    end
end
function (c::StopWhenResidualIsReducedByFactorOrPower)(
    mp::AbstractManoptProblem, tcgstate::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    TpM = get_manifold(mp)
    M = base_manifold(TpM)
    p = TpM.point
    if norm(M, p, tcgstate.residual) <=
       tcgstate.initialResidualNorm * min(c.κ, tcgstate.initialResidualNorm^(c.θ)) && i > 0
        c.reason = "The norm of the residual is less than or equal either to κ=$(c.κ) times the norm of the initial residual or to the norm of the initial residual to the power 1 + θ=$(1+(c.θ)). \n"
        return true
    end
    return false
end
function status_summary(c::StopWhenResidualIsReducedByFactorOrPower)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Residual reduced by factor $(c.κ) or power $(c.θ):\t$s"
end
function show(io::IO, c::StopWhenResidualIsReducedByFactorOrPower)
    return print(
        io,
        "StopWhenResidualIsReducedByFactorOrPower($(c.κ), $(c.θ))\n    $(status_summary(c))",
    )
end

@doc raw"""
    update_stopping_criterion!(c::StopWhenResidualIsReducedByFactorOrPower, :ResidualPower, v)
Update the residual Power `θ`  to `v`.
"""
function update_stopping_criterion!(
    c::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualPower}, v
)
    c.θ = v
    return c
end

@doc raw"""
    update_stopping_criterion!(c::StopWhenResidualIsReducedByFactorOrPower, :ResidualFactor, v)
Update the residual Factor `κ` to `v`.
"""
function update_stopping_criterion!(
    c::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualFactor}, v
)
    c.κ = v
    return c
end

@doc raw"""
    StopWhenTrustRegionIsExceeded <: StoppingCriterion

A functor for testing if the norm of the next iterate in the Steihaug-Toint truncated conjugate gradient
method is larger than the trust-region radius ``θ \leq \Vert Y_{k}^{*} \Vert_p``
and to end the algorithm when the trust region has been left.

# Fields

* `reason`: stores a reason of stopping if the stopping criterion has been reached, see [`get_reason`](@ref).

# Constructor

    StopWhenTrustRegionIsExceeded()

initialize the StopWhenTrustRegionIsExceeded functor to indicate to stop after
the norm of the next iterate is greater than the trust-region radius.

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenTrustRegionIsExceeded <: StoppingCriterion
    reason::String
    at_iteration::Int
end
StopWhenTrustRegionIsExceeded() = StopWhenTrustRegionIsExceeded("", 0)
function (c::StopWhenTrustRegionIsExceeded)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if tcgs.YPY >= tcgs.trust_region_radius^2 && i >= 0
        c.reason = "Trust-region radius violation (‖Y‖² = $(tcgs.YPY)) >= $(tcgs.trust_region_radius^2) = trust_region_radius²). \n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenTrustRegionIsExceeded)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Trust region exceeded:\t$s"
end
function show(io::IO, c::StopWhenTrustRegionIsExceeded)
    return print(io, "StopWhenTrustRegionIsExceeded()\n    $(status_summary(c))")
end

@doc raw"""
    StopWhenCurvatureIsNegative <: StoppingCriterion

A functor for testing if the curvature of the model is negative,
``⟨δ_k, \operatorname{Hess}[F](\delta_k)⟩_p ≦ 0``.
In this case, the model is not strictly convex, and the stepsize as computed does not
yield a reduction of the model.

# Fields
* `reason`: stores a reason of stopping if the stopping criterion has been reached,
  see [`get_reason`](@ref).

# Constructor

    StopWhenCurvatureIsNegative()

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenCurvatureIsNegative <: StoppingCriterion
    reason::String
    at_iteration::Int
end
StopWhenCurvatureIsNegative() = StopWhenCurvatureIsNegative("", 0)
function (c::StopWhenCurvatureIsNegative)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if tcgs.δHδ <= 0 && i > 0
        c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(tcgs.δHδ))) <= 0).\n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenCurvatureIsNegative)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Curvature is negative:\t$s"
end
function show(io::IO, c::StopWhenCurvatureIsNegative)
    return print(io, "StopWhenCurvatureIsNegative()\n    $(status_summary(c))")
end

@doc raw"""
    StopWhenModelIncreased <: StoppingCriterion

A functor for testing if the curvature of the model value increased.

# Fields
* `reason`: stores a reason of stopping if the stopping criterion has been reached,
  see [`get_reason`](@ref).

# Constructor

    StopWhenModelIncreased()

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenModelIncreased <: StoppingCriterion
    reason::String
    at_iteration::Int
    model_value::Float64
end
StopWhenModelIncreased() = StopWhenModelIncreased("", 0, Inf)
function (c::StopWhenModelIncreased)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
        c.model_value = Inf
    end
    if i > 0 && (tcgs.model_value > c.model_value)
        c.reason = "Model value increased from $(c.model_value) to $(tcgs.model_value).\n"
        return true
    end
    c.model_value = tcgs.model_value
    return false
end
function status_summary(c::StopWhenModelIncreased)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Model Increased:\t$s"
end
function show(io::IO, c::StopWhenModelIncreased)
    return print(io, "StopWhenModelIncreased()\n    $(status_summary(c))")
end

@doc raw"""
    truncated_conjugate_gradient_descent(M, f, grad_f, p; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, p, X; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, Hess_f; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, Hess_f, p; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, Hess_f, p, X; kwargs...)
    truncated_conjugate_gradient_descent(M, mho::ManifoldHessianObjective, p, X; kwargs...)
    truncated_conjugate_gradient_descent(M, trmo::TrustRegionModelObjective, p, X; kwargs...)

solve the trust-region subproblem

```math
\begin{align*}
\operatorname*{arg\,min}_{Y  ∈  T_p\mathcal{M}}&\ m_p(Y) = f(p) +
⟨\operatorname{grad}f(p), Y⟩_p + \frac{1}{2} ⟨\mathcal{H}_p[Y], Y⟩_p\\
\text{such that}& \ \lVert Y \rVert_p ≤ Δ
\end{align*}
```

on a manifold M by using the Steihaug-Toint truncated conjugate-gradient (tCG) method.
For a description of the algorithm and theorems offering convergence guarantees,
see [AbsilBakerGallivan:2006, ConnGouldToint:2000](@cite).

# Input

* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``f: \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of `F`
* `Hess_f`: (optional, cf. [`ApproxHessianFiniteDifference`](@ref)) the Hessian ``\operatorname{Hess}f: T_p\mathcal M → T_p\mathcal M``, ``X ↦ \operatorname{Hess}F(p)[X] = ∇_X\operatorname{grad}f(p)``
* `p`:      a point on the manifold ``p ∈ \mathcal M``
* `X`:      an initial tangential vector ``X ∈ T_p\mathcal M``

Instead of the three functions, you either provide a [`ManifoldHessianObjective`](@ref) `mho`
which is then used to build the trust region model, or a [`TrustRegionModelObjective`](@ref) `trmo`
directly.

# Optional

* `evaluation`:          ([`AllocatingEvaluation`](@ref)) specify whether the gradient and Hessian work by
   allocation (default) or [`InplaceEvaluation`](@ref) in place
* `preconditioner`:      a preconditioner for the Hessian H
* `θ`:                   (`1.0`) 1+θ is the superlinear convergence target rate.
* `κ`:                   (`0.1`) the linear convergence target rate.
* `randomize`:           set to true if the trust-region solve is initialized to a random tangent vector.
  This disables preconditioning.
* `trust_region_radius`: (`injectivity_radius(M)/4`) a trust-region radius
* `project!`:            (`copyto!`) for numerical stability it is possible to project onto
  the tangent space after every iteration. By default this only copies instead.
* `stopping_criterion`:  ([`StopAfterIteration`](@ref)`(manifol_dimension(M)) | `[`StopWhenResidualIsReducedByFactorOrPower`](@ref)`(;κ=κ, θ=θ) | `[`StopWhenCurvatureIsNegative`](@ref)`() | `[`StopWhenTrustRegionIsExceeded`](@ref)`() | `[`StopWhenModelIncreased`](@ref)`()`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop,

and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``Y^*``, see [`get_solver_return`](@ref) for details

# See also
[`trust_regions`](@ref)
"""
truncated_conjugate_gradient_descent(M::AbstractManifold, args; kwargs...)
# No Hessian, no point/vector
function truncated_conjugate_gradient_descent(M::AbstractManifold, f, grad_f; kwargs...)
    return truncated_conjugate_gradient_descent(M, f, grad_f, rand(M); kwargs...)
end
# No Hessian, no vector
function truncated_conjugate_gradient_descent(M::AbstractManifold, f, grad_f, p; kwargs...)
    return truncated_conjugate_gradient_descent(
        M, f, grad_f, p, rand(M; vector_at=p); kwargs...
    )
end
# no Hessian
function truncated_conjugate_gradient_descent(
    M::AbstractManifold,
    f,
    grad_f,
    p,
    X;
    evaluation=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
)
    Hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return truncated_conjugate_gradient_descent(
        M,
        f,
        grad_f,
        Hess_f,
        p,
        X;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
# no point/vector
function truncated_conjugate_gradient_descent(
    M::AbstractManifold, f, grad_f, Hess_f::TH; kwargs...
) where {TH<:Function}
    return truncated_conjugate_gradient_descent(M, f, grad_f, Hess_f, rand(M); kwargs...)
end
# no vector
function truncated_conjugate_gradient_descent(
    M::AbstractManifold, f, grad_f, Hess_f::TH, p; kwargs...
) where {TH<:Function}
    return truncated_conjugate_gradient_descent(
        M, f, grad_f, Hess_f, p, rand(M; vector_at=p); kwargs...
    )
end
#
# All defaults filled, generate mho
#
function truncated_conjugate_gradient_descent(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH,
    p,
    X;
    evaluation=AllocatingEvaluation(),
    preconditioner=if evaluation isa InplaceEvaluation
        (M, Y, p, X) -> (Y .= X)
    else
        (M, p, X) -> X
    end,
    kwargs...,
) where {TH<:Function}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f, preconditioner; evaluation=evaluation)
    return truncated_conjugate_gradient_descent(
        M, mho, p, X; evaluation=evaluation, kwargs...
    )
end
function truncated_conjugate_gradient_descent(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH, #fill a default below before dispatching on p::Number
    p::Number,
    X::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    preconditioner=(M, p, X) -> X,
    kwargs...,
) where {TH<:Function}
    q = [p]
    Y = [X]
    f_(M, p) = f(M, p[])
    if evaluation isa AllocatingEvaluation
        grad_f_ = (M, p) -> [grad_f(M, p[])]
        Hess_f_ = (M, p, X) -> [Hess_f(M, p[], X[])]
        precon_ = (M, p, X) -> [preconditioner(M, p[], X[])]
    else
        grad_f_ = (M, X, p) -> (X .= [grad_f(M, p[])])
        Hess_f_ = (M, Y, p, X) -> (Y .= [Hess_f(M, p[], X[])])
        precon_ = (M, Y, p, X) -> (Y .= [preconditioner(M, p[], X[])])
    end
    rs = truncated_conjugate_gradient_descent(
        M,
        f_,
        grad_f_,
        Hess_f_,
        q,
        Y;
        preconditioner=precon_,
        evaluation=evaluation,
        kwargs...,
    )
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
#
# Objective 1 -> generate model
function truncated_conjugate_gradient_descent(
    M::AbstractManifold, mho::O, p, X; kwargs...
) where {O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    trmo = TrustRegionModelObjective(mho)
    TpM = TangentSpace(M, copy(M, p))
    return truncated_conjugate_gradient_descent(TpM, trmo, p, X; kwargs...)
end
#
# Objective 2, a tangent space model -> Allocate and call !
function truncated_conjugate_gradient_descent(
    M::AbstractManifold, mho::O, p, X; kwargs...
) where {
    O<:Union{
        AbstractManifoldSubObjective,
        AbstractDecoratedManifoldObjective{E,<:AbstractManifoldSubObjective} where E,
    },
}
    q = copy(M, p)
    Y = copy(M, p, X)
    return truncated_conjugate_gradient_descent!(M, mho, q, Y; kwargs...)
end

@doc raw"""
    truncated_conjugate_gradient_descent!(M, f, grad_f, Hess_f, p, X; kwargs...)
    truncated_conjugate_gradient_descent!(M, f, grad_f, p, X; kwargs...)

solve the trust-region subproblem in place of `X` (and `p`).

# Input
* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of `f`
* `Hess_f`: the Hessian ``\operatorname{Hess}f(x): T_p\mathcal M → T_p\mathcal M``, ``X ↦ \operatorname{Hess}f(p)[X]``
* `p`:      a point on the manifold ``p ∈ \mathcal M``
* `X`:      an update tangential vector ``X ∈ T_x\mathcal M``

For more details and all optional arguments, see [`truncated_conjugate_gradient_descent`](@ref).
"""
truncated_conjugate_gradient_descent!(M::AbstractManifold, args...; kwargs...)
# no Hessian
function truncated_conjugate_gradient_descent!(
    M::AbstractManifold,
    f,
    grad_f,
    p,
    X;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
)
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return truncated_conjugate_gradient_descent!(
        M,
        f,
        grad_f,
        hess_f,
        p,
        X;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
# From functions to objective
function truncated_conjugate_gradient_descent!(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH,
    p,
    X;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    preconditioner=if evaluation isa InplaceEvaluation
        (M, Y, p, X) -> (Y .= X)
    else
        (M, p, X) -> X
    end,
    kwargs...,
) where {TH<:Function}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f, preconditioner; evaluation=evaluation)
    return truncated_conjugate_gradient_descent!(
        M, mho, p, X; evaluation=evaluation, kwargs...
    )
end
function truncated_conjugate_gradient_descent!(
    M::AbstractManifold, mho::O, p, X; kwargs...
) where {O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    trmo = TrustRegionModelObjective(mho)
    TpM = TangentSpace(M, copy(M, p))
    return truncated_conjugate_gradient_descent!(TpM, trmo, p, X; kwargs...)
end
function truncated_conjugate_gradient_descent!(
    TpM::TangentSpace,
    trm::TrustRegionModelObjective,
    p,
    X;
    trust_region_radius::Float64=injectivity_radius(TpM) / 4,
    θ::Float64=1.0,
    κ::Float64=0.1,
    randomize::Bool=false,
    stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(TpM)) |
                                          StopWhenResidualIsReducedByFactorOrPower(;
                                              κ=κ, θ=θ
                                          ) |
                                          StopWhenTrustRegionIsExceeded() |
                                          StopWhenCurvatureIsNegative() |
                                          StopWhenModelIncreased(),
    project!::Proj=copyto!,
    kwargs..., #collect rest
) where {Proj}
    dtrm = decorate_objective!(TpM, trm; kwargs...)
    mp = DefaultManoptProblem(TpM, dtrm)
    tcgs = TruncatedConjugateGradientState(
        TpM,
        X;
        trust_region_radius=trust_region_radius,
        randomize=randomize,
        θ=θ,
        κ=κ,
        stopping_criterion=stopping_criterion,
        (project!)=project!,
    )
    dtcgs = decorate_state!(tcgs; kwargs...)
    solve!(mp, dtcgs)
    return get_solver_return(get_objective(mp), dtcgs)
end

#
# Maybe these could be improved a bit in readability some time
#
function initialize_solver!(
    mp::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState
)
    TpM = get_manifold(mp)
    M = base_manifold(TpM)
    p = TpM.point
    trmo = get_objective(mp)
    # TODO Reworked until here
    (tcgs.randomize) || zero_vector!(M, tcgs.Y, p)
    tcgs.HY = tcgs.randomize ? get_objective_hessian(M, trmo, p, tcgs.Y) : zero_vector(M, p)
    tcgs.X = get_objective_gradient(M, trmo, p) # Initialize gradient
    tcgs.residual = tcgs.randomize ? tcgs.X + tcgs.HY : tcgs.X
    tcgs.z = if tcgs.randomize
        tcgs.residual
    else
        get_objective_preconditioner(M, trmo, p, tcgs.residual)
    end
    tcgs.δ = -copy(M, p, tcgs.z)
    tcgs.Hδ = zero_vector(M, p)
    tcgs.δHδ = real(inner(M, p, tcgs.δ, tcgs.Hδ))
    tcgs.YPδ = tcgs.randomize ? real(inner(M, p, tcgs.Y, tcgs.δ)) : zero(tcgs.δHδ)
    tcgs.δPδ = real(inner(M, p, tcgs.residual, tcgs.z))
    tcgs.YPY = tcgs.randomize ? real(inner(M, p, tcgs.Y, tcgs.Y)) : zero(tcgs.δHδ)
    if tcgs.randomize
        tcgs.model_value =
            real(inner(M, p, tcgs.Y, tcgs.X)) + 0.5 * real(inner(M, p, tcgs.Y, tcgs.HY))
    else
        tcgs.model_value = 0
    end
    tcgs.z_r = real(inner(M, p, tcgs.z, tcgs.residual))
    tcgs.initialResidualNorm = norm(M, p, tcgs.residual)
    return tcgs
end
function step_solver!(
    mp::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, ::Any
)
    TpM = get_manifold(mp)
    M = base_manifold(TpM)
    p = TpM.point
    trmo = get_objective(mp)
    get_objective_hessian!(M, tcgs.Hδ, trmo, p, tcgs.δ)
    tcgs.δHδ = real(inner(M, p, tcgs.δ, tcgs.Hδ))
    α = tcgs.z_r / tcgs.δHδ
    YPY_new = tcgs.YPY + 2 * α * tcgs.YPδ + α^2 * tcgs.δPδ
    # Check against negative curvature and trust-region radius violation.
    if tcgs.δHδ <= 0 || YPY_new >= tcgs.trust_region_radius^2
        τ =
            (
                -tcgs.YPδ +
                sqrt(tcgs.YPδ^2 + tcgs.δPδ * (tcgs.trust_region_radius^2 - tcgs.YPY))
            ) / tcgs.δPδ
        copyto!(M, tcgs.Y, p, tcgs.Y + τ * tcgs.δ)
        copyto!(M, tcgs.HY, p, tcgs.HY + τ * tcgs.Hδ)
        tcgs.YPY = YPY_new
        return tcgs
    end
    tcgs.YPY = YPY_new
    new_Y = tcgs.Y + α * tcgs.δ # Update iterate Y
    new_HY = tcgs.HY + α * tcgs.Hδ # Update HY
    new_model_value =
        real(inner(M, p, new_Y, tcgs.X)) + 0.5 * real(inner(M, p, new_Y, new_HY))
    # If model was not improved with this iterate -> end iteration
    if new_model_value >= tcgs.model_value
        tcgs.model_value = new_model_value
        return tcgs
    end
    # otherweise accept step
    copyto!(M, tcgs.Y, p, new_Y)
    tcgs.model_value = new_model_value
    copyto!(M, tcgs.HY, p, new_HY)
    tcgs.residual = tcgs.residual + α * tcgs.Hδ

    # Precondition the residual if not running in random mode
    tcgs.z = if tcgs.randomize
        tcgs.residual
    else
        get_objective_preconditioner(M, trmo, p, tcgs.residual)
    end
    zr = real(inner(M, p, tcgs.z, tcgs.residual))
    # Compute new search direction.
    β = zr / tcgs.z_r
    tcgs.z_r = zr
    tcgs.δ = -tcgs.z + β * tcgs.δ
    # potentially stabilize step by projecting.
    tcgs.project!(M, tcgs.δ, p, tcgs.δ)
    tcgs.YPδ = β * (α * tcgs.δPδ + tcgs.YPδ)
    tcgs.δPδ = tcgs.z_r + β^2 * tcgs.δPδ
    return tcgs
end
get_solver_result(s::TruncatedConjugateGradientState) = s.Y
