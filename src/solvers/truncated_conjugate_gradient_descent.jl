@doc """
    TruncatedConjugateGradientState <: AbstractHessianSolverState

describe the Steihaug-Toint truncated conjugate-gradient method, with

# Fields

Let `T` denote the type of a tangent vector and `R <: Real`.

* `δ::T`:                     the conjugate gradient search direction
* `δHδ`, `YPδ`, `δPδ`, `YPδ`: temporary inner products with `Hδ` and preconditioned inner products.
* `Hδ`, `HY`:                 temporary results of the Hessian applied to `δ` and `Y`, respectively.
* `κ::R`:                     the linear convergence target rate.
* `project!=copyto!`: for numerical stability it is possible to project onto the tangent space after every iteration.
  the function has to work inplace of `Y`, that is `(M, Y, p, X) -> Y`, where `X` and `Y` can be the same memory.
* `randomize`:          indicate whether `X` is initialised to a random vector or not
* `residual::T`:                 the gradient of the model ``m(Y)``
* $(_field_stop)
* `θ::R`:                     the superlinear convergence target rate of ``1+θ``
* `trust_region_radius::R`:   the trust-region radius
* `X::T`:                     the gradient ``$(_l_grad)f(p)``
* `Y::T`:                     current iterate tangent vector
* `z::T`:                     the preconditioned residual
* `z_r::R`:                   inner product of the residual and `z`

# Constructor

    TruncatedConjugateGradientState(TpM::TangentSpace, Y=rand(TpM); kwargs...)

Initialise the TCG state.

## Input

* `TpM`: a [`TangentSpace`](@extref `ManifoldsBase.TangentSpace`)
* `Y`: an initial tangent vector

## Keyword arguments

* `κ=0.1`
* `project!::F=copyto!`: initialise the numerical stabilisation to just copy the result
* `randomize=false`
* `θ=1.0`
* `trust_region_radius=`[`injectivity_radius`](@extref `ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}`)`(base_manifold(TpM)) / 4`
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(`[`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`)`(base_manifold(TpM))`)
  $(_sc_any)[`StopWhenResidualIsReducedByFactorOrPower`](@ref)`(; κ=κ, θ=θ)`$(_sc_any)[`StopWhenTrustRegionIsExceeded`](@ref)`()`
  $(_sc_any)[`StopWhenCurvatureIsNegative`](@ref)`()`$(_sc_any)[`StopWhenModelIncreased`](@ref)`()`:
  $(_kw_stopping_criterion)

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
@doc """
    StopWhenResidualIsReducedByFactorOrPower <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
either by a power of 1+θ or by a factor κ compared to the norm of the initial
residual. The criterion hence reads

``$(_l_norm("r_k","p")) ≦ $(_l_norm("r_0","p^{(0)}")) \\min \\bigl( κ, $(_l_norm("r_0","p^{(0)}"))  \\bigr)``.

# Fields

* `κ`:      the reduction factor
* `θ`:      part of the reduction power
* $(_field_at_iteration)

# Constructor

    StopWhenResidualIsReducedByFactorOrPower(; κ=0.1, θ=1.0)

Initialize the StopWhenResidualIsReducedByFactorOrPower functor to indicate to stop after
the norm of the current residual is lesser than either the norm of the initial residual
to the power of 1+θ or the norm of the initial residual times κ.

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenResidualIsReducedByFactorOrPower{F} <: StoppingCriterion
    κ::F
    θ::F
    at_iteration::Int
    function StopWhenResidualIsReducedByFactorOrPower(; κ::F=0.1, θ::F=1.0) where {F<:Real}
        return new{F}(κ, θ, -1)
    end
end
function (c::StopWhenResidualIsReducedByFactorOrPower)(
    mp::AbstractManoptProblem, tcgstate::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.at_iteration = -1
    end
    TpM = get_manifold(mp)
    M = base_manifold(TpM)
    p = TpM.point
    if norm(M, p, tcgstate.residual) <=
       tcgstate.initialResidualNorm * min(c.κ, tcgstate.initialResidualNorm^(c.θ)) && i > 0
        c.at_iteration = i
        return true
    end
    return false
end
function get_reason(c::StopWhenResidualIsReducedByFactorOrPower)
    if (c.at_iteration >= 0)
        return "The norm of the residual is less than or equal either to κ=$(c.κ) times the norm of the initial residual or to the norm of the initial residual to the power 1 + θ=$(1+(c.θ)). \n"
    end
    return ""
end
function status_summary(c::StopWhenResidualIsReducedByFactorOrPower)
    has_stopped = (c.at_iteration >= 0)
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

@doc """
    StopWhenTrustRegionIsExceeded <: StoppingCriterion

A functor for testing if the norm of the next iterate in the Steihaug-Toint truncated conjugate gradient
method is larger than the trust-region radius ``θ ≤ $(_l_norm("Y^{(k)}^{*}","p^{(k)}"))``
and to end the algorithm when the trust region has been left.

# Fields

* $(_field_at_iteration)
* `trr` the trust region radius
* `YPY` the computed norm of ``Y``.

# Constructor

    StopWhenTrustRegionIsExceeded()

initialize the StopWhenTrustRegionIsExceeded functor to indicate to stop after
the norm of the next iterate is greater than the trust-region radius.

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenTrustRegionIsExceeded{F} <: StoppingCriterion
    at_iteration::Int
    YPY::F
    trr::F
    StopWhenTrustRegionIsExceeded(v::F) where {F} = new{F}(-1, zero(v), zero(v))
end
StopWhenTrustRegionIsExceeded() = StopWhenTrustRegionIsExceeded(0.0)
function (c::StopWhenTrustRegionIsExceeded)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.at_iteration = -1
    end
    if tcgs.YPY >= tcgs.trust_region_radius^2 && i >= 0
        c.YPY = tcgs.YPY
        c.trr = tcgs.trust_region_radius
        c.at_iteration = i
        return true
    end
    return false
end
function get_reason(c::StopWhenTrustRegionIsExceeded)
    if c.at_iteration >= 0
        return "Trust-region radius violation (‖Y‖² = $(c.YPY)) >= $(c.trr^2) = trust_region_radius²). \n"
    end
    return ""
end
function status_summary(c::StopWhenTrustRegionIsExceeded)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Trust region exceeded:\t$s"
end
function show(io::IO, c::StopWhenTrustRegionIsExceeded)
    return print(io, "StopWhenTrustRegionIsExceeded()\n    $(status_summary(c))")
end

@doc """
    StopWhenCurvatureIsNegative <: StoppingCriterion

A functor for testing if the curvature of the model is negative,
``⟨δ_k, $(_l_Hess) F(p)[δ_k]⟩_p ≦ 0``.
In this case, the model is not strictly convex, and the stepsize as computed does not
yield a reduction of the model.

# Fields

* $(_field_at_iteration)
* `value` store the value of the inner product.
* `reason`: stores a reason of stopping if the stopping criterion has been reached,
  see [`get_reason`](@ref).

# Constructor

    StopWhenCurvatureIsNegative()

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenCurvatureIsNegative{R} <: StoppingCriterion
    value::R
    at_iteration::Int
end
StopWhenCurvatureIsNegative() = StopWhenCurvatureIsNegative(0.0)
StopWhenCurvatureIsNegative(v::R) where {R<:Real} = StopWhenCurvatureIsNegative{R}(v, -1)
function (c::StopWhenCurvatureIsNegative)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.at_iteration = -1
    end
    if tcgs.δHδ <= 0 && i > 0
        c.value = tcgs.δHδ
        c.at_iteration = i
        return true
    end
    return false
end
function get_reason(c::StopWhenCurvatureIsNegative)
    if c.at_iteration >= 0
        return "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(c.value))) <= 0).\n"
    end
    return ""
end
function status_summary(c::StopWhenCurvatureIsNegative)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Curvature is negative:\t$s"
end
function show(io::IO, c::StopWhenCurvatureIsNegative)
    return print(io, "StopWhenCurvatureIsNegative()\n    $(status_summary(c))")
end

@doc """
    StopWhenModelIncreased <: StoppingCriterion

A functor for testing if the curvature of the model value increased.

# Fields

* $(_field_at_iteration)
* `model_value`stre the last model value
* `inc_model_value` store the model value that increased

# Constructor

    StopWhenModelIncreased()

# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenModelIncreased{F} <: StoppingCriterion
    at_iteration::Int
    model_value::F
    inc_model_value::F
end
StopWhenModelIncreased() = StopWhenModelIncreased(-1, Inf, Inf)
function (c::StopWhenModelIncreased)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.at_iteration = -1
        c.model_value = Inf
        c.inc_model_value = Inf
    end
    if i > 0 && (tcgs.model_value > c.model_value)
        c.inc_model_value = tcgs.model_value
        c.at_iteration = i
        return true
    end
    c.model_value = tcgs.model_value
    return false
end
function get_reason(c::StopWhenModelIncreased)
    if c.at_iteration >= 0
        return "Model value increased from $(c.model_value) to $( c.inc_model_value).\n"
    end
    return ""
end
function status_summary(c::StopWhenModelIncreased)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Model Increased:\t$s"
end
function show(io::IO, c::StopWhenModelIncreased)
    return print(io, "StopWhenModelIncreased()\n    $(status_summary(c))")
end

_doc_TCG_subproblem = raw"""
```math
\begin{align*}
\operatorname*{arg\,min}_{Y  ∈  T_p\mathcal{M}}&\ m_p(Y) = f(p) +
⟨\operatorname{grad}f(p), Y⟩_p + \frac{1}{2} ⟨\mathcal{H}_p[Y], Y⟩_p\\
\text{such that}& \ \lVert Y \rVert_p ≤ Δ
\end{align*}
```
"""
_doc_TCGD = """
    truncated_conjugate_gradient_descent(M, f, grad_f, p; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, p, X; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, Hess_f; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, Hess_f, p; kwargs...)
    truncated_conjugate_gradient_descent(M, f, grad_f, Hess_f, p, X; kwargs...)
    truncated_conjugate_gradient_descent(M, mho::ManifoldHessianObjective, p, X; kwargs...)
    truncated_conjugate_gradient_descent(M, trmo::TrustRegionModelObjective, p, X; kwargs...)
    truncated_conjugate_gradient_descent!(M, f, grad_f, Hess_f, p, X; kwargs...)
    truncated_conjugate_gradient_descent!(M, f, grad_f, p, X; kwargs...)
    truncated_conjugate_gradient_descent!(M, mho::ManifoldHessianObjective, p, X; kwargs...)
    truncated_conjugate_gradient_descent!(M, trmo::TrustRegionModelObjective, p, X; kwargs...)

solve the trust-region subproblem

$(_doc_TCG_subproblem)

on a manifold ``$(_l_M)`` by using the Steihaug-Toint truncated conjugate-gradient (tCG) method.
This can be done inplace of `X`.

For a description of the algorithm and theorems offering convergence guarantees,
see [AbsilBakerGallivan:2006, ConnGouldToint:2000](@cite).

# Input

$(_arg_M)
$(_arg_f)
$(_arg_grad_f)
$(_arg_Hess_f)
$(_arg_p)
$(_arg_X)

Instead of the three functions, you either provide a [`ManifoldHessianObjective`](@ref) `mho`
which is then used to build the trust region model, or a [`TrustRegionModelObjective`](@ref) `trmo`
directly.

# Keyword arguments

* $(_kw_evaluation_default): $(_kw_evaluation)
* `preconditioner`:       a preconditioner for the Hessian H.
  This is either an allocating function `(M, p, X) -> Y` or an in-place function `(M, Y, p, X) -> Y`,
  see `evaluation`, and by default set to the identity.
* `θ=1.0`:                the superlinear convergence target rate of ``1+θ``
* `κ=0.1`:                the linear convergence target rate.
* `project!=copyto!`: for numerical stability it is possible to project onto the tangent space after every iteration.
  the function has to work inplace of `Y`, that is `(M, Y, p, X) -> Y`, where `X` and `Y` can be the same memory.
* `randomize`:            indicate whether `X` is initialised to a random vector or not.
  This disables preconditioning.
* $(_kw_retraction_method_default): $(_kw_retraction_method)
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(`[`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`)`(base_manifold(TpM))`)
  $(_sc_any)[`StopWhenResidualIsReducedByFactorOrPower`](@ref)`(; κ=κ, θ=θ)`$(_sc_any)[`StopWhenTrustRegionIsExceeded`](@ref)`()`
  $(_sc_any)[`StopWhenCurvatureIsNegative`](@ref)`()`$(_sc_any)[`StopWhenModelIncreased`](@ref)`()`:
  $(_kw_stopping_criterion)
* `trust_region_radius=`[`injectivity_radius`](@extref `ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}`)`(M) / 4`: the initial trust-region radius

$(_kw_others)

$(_doc_sec_output)

# See also

[`trust_regions`](@ref)
"""

@doc "$(_doc_TCGD)"
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

@doc "$(_doc_TCGD)"
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

function initialize_solver!(
    mp::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState
)
    TpM = get_manifold(mp)
    M = base_manifold(TpM)
    p = TpM.point
    trmo = get_objective(mp)
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
    # otherwise accept step
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
