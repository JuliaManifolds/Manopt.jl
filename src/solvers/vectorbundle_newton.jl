@doc """
    VectorBundleNewtonState{P,T} <: AbstractManoptSolverState

Is state for the vector bundle Newton method

# Fields

$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:p; name = "p_trial"))
  next iterate needed for simplified Newton
$(_fields(:X))
  as current Newton direction
$(_fields(:sub_problem))
  currently only the closed form solution is implemented, that is, this is a functor that maps
  either `(problem::`[`VectorBundleManoptProblem`](@ref)`, state::VectorBundleNewtonState) -> X` or `(problem, X, state) -> X` to compute the Newton direction.
$(_fields(:sub_state)) specify how the sub_problem is evaluated, e.g. [`AllocatingEvaluation`](@ref) or [`InplaceEvaluation`](@ref)
$(_fields(:stopping_criterion; name = "stop"))
$(_fields([:stepsize, :retraction_method]))

# Constructor

    VectorBundleNewtonState(M, E, p, sub_problem, sub_state; kwargs...)

# Input

$(_args(:M))
* `E`: range vector bundle
$(_args([:p, :sub_state, :sub_problem]))

# Keyword arguments

$(_kwargs(:X; add_properties = [:as_Memory]))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`VectorBundleNewtonState`](@ref)`)"))
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
"""
mutable struct VectorBundleNewtonState{
        P, T, Pr, St,
        TStop <: StoppingCriterion, TStep <: Stepsize, TRTM <: AbstractRetractionMethod,
    } <: AbstractGradientSolverState
    p::P
    p_trial::P
    X::T
    sub_problem::Pr
    sub_state::St
    stop::TStop
    stepsize::TStep
    retraction_method::TRTM
end

function VectorBundleNewtonState(
        M::AbstractManifold, E::AbstractManifold, p::P, sub_problem::Pr, sub_state::Op;
        X::T = zero_vector(M, p),
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        stopping_criterion::SC = StopAfterIteration(1000),
        stepsize::S = default_stepsize(M, VectorBundleNewtonState)
    ) where {
        P, T, Pr, Op, RM <: AbstractRetractionMethod, SC <: StoppingCriterion, S <: Stepsize,
    }
    return VectorBundleNewtonState{P, T, Pr, Op, SC, S, RM}(
        p, copy(M, p), X,
        sub_problem, sub_state, stopping_criterion, stepsize, retraction_method
    )
end

function Base.show(io::IO, vbns::VectorBundleNewtonState)
    print(io, "VectorBundleNewtonState(M, E, p, $(vbns.sub_problem), $(vbns.sub_state);\n$(_MANOPT_INDENT)")
    print(io, "retraction_method = $(vbns.retraction_method),\n$(_MANOPT_INDENT)")
    print(io, "stopping_criterion = $(status_summary(vbns.stop; context = :short)),\n$(_MANOPT_INDENT)")
    print(io, "stepsize = $(vbns.stepsize),\n$(_MANOPT_INDENT)")
    print(io, "X = $(vbns.X),\n")
    return print(io, ")")
end

@doc """
AffineCovariantStepsize <: Stepsize

A functor to provide an affine covariant stepsize generalizing the idea of following Newton paths introduced by [WeiglBergmannSchiela:2025; Section 4](@cite).
It can be used to derive a damped Newton method. The stepsizes (damping factors) are computed
by a predictor-corrector-loop using an affine covariant quantity ``θ`` to measure local convergence.

# Fields

* `α`:             stepsize (damping factor)
* `θ`:             quantity that measures local convergence of Newton's method
* `θ_des`:         desired θ
* `θ_acc`:         acceptable θ
* `last_stepsize`: last computed stepsize (this is an auxiliary variable used within the algorithm)
* `outer_norm`:    if `M` is a manifold with components, this is used to specify the norm,
  that is used to compute the overall distance based on the element-wise distance.

# Example

On an $(_link(:AbstractPowerManifold)) like ``$(_math(:Manifold))nifold))nifold))) = $(_math(:Manifold; M = "N"))^n``
any point ``p = (p_1,…,p_n) ∈ $(_math(:Manifold))nifold)))`` is a vector of length ``n`` with of points ``p_i ∈ $(_math(:Manifold; M = "N"))``.
Then, denoting the `outer_norm` by ``r``, the distance of two points ``p,q ∈ $(_math(:Manifold)))``
is given by

```math
$(_math(:distance))(p,q) = $(_tex(:Bigl))( $(_tex(:sum))_{k=1}^n $(_math(:distance))(p_k,q_k)^r $(_tex(:Bigr)))^{$(_tex(:frac, "1", "r"))},
```

where the sum turns into a maximum for the case ``r=∞``.
The `outer_norm` has no effect on manifolds that do not consist of components.

If the manifold does not have components, the outer norm is ignored.

# Constructor

    AffineCovariantStepsize(
        M::AbstractManifold=DefaultManifold(2);
        α=1.0, θ=1.3, θ_des=0.5, θ_acc=1.1*θ_des, outer_norm::Real=missing
    )

Initializes all fields, where none of them is mandatory. The length is set to ``1.0``.

Since the computation of the convergence monitor ``θ`` requires simplified Newton directions a method for computing them has to be provided.
This should be implemented as a method of the `newton_equation(M, VB, p, p_trial)` as parameters and returning a representation of the (transported) ``F(p_{$(_tex(:rm, "trial"))})``.
"""
mutable struct AffineCovariantStepsize{R <: Real, N <: Union{Real, Missing}} <: Stepsize
    α::R
    θ::R
    θ_des::R
    θ_acc::R
    last_stepsize::R
    outer_norm::N
end
function AffineCovariantStepsize(
        ::AbstractManifold = DefaultManifold(2);
        α::Real = 1.0, θ::Real = 1.3, θ_des::Real = 0.5, θ_acc::Real = 1.1 * θ_des, outer_norm::N = missing
    ) where {N <: Union{Real, Missing}}
    R = promote_type(typeof(α), typeof(θ), typeof(θ_des), typeof(θ_acc))
    return AffineCovariantStepsize{R, N}(
        convert(R, α), convert(R, θ), convert(R, θ_des), convert(R, θ_acc), convert(R, 1.0), outer_norm
    )
end
function Base.show(io::IO, acs::AffineCovariantStepsize)
    print(io, "AffineCovariantStepsize(; α = ", acs.α, ", θ = ", acs.θ, ", θ_des = ", acs.θ_des)
    print(io, ", θ_acc = ", acs.θ_acc)
    !(ismissing(acs.outer_norm)) && print(io, ", outer_norm = ", acs.outer_norm)
    return print(io, ")")
end
function status_summary(acs::AffineCovariantStepsize; context = :default)
    (context === :short) && repr(acs)
    (context === :inline) && return "An affine covariant step size (last step size: $(acs.last_stepsize))"
    on = ismissing(acs.outer_norm) ? "" : "\n* outer norm:       $(_MANOPT_INDENT)$(acs.outer_norm)"
    return """
    An affine covariant step size
    (last step size: $(acs.last_stepsize))

    ## Parameters
    * damping factor α: $(_MANOPT_INDENT)$(acs.α)
    * θ:                $(_MANOPT_INDENT)$(acs.θ)
    * desired θ:        $(_MANOPT_INDENT)$(acs.θ_des)
    * acceptable θ:     $(_MANOPT_INDENT)$(acs.θ_acc)$(on)
    """
end
function (acs::AffineCovariantStepsize)(
        amp::AbstractManoptProblem, ams::VectorBundleNewtonState, ::Any, args...; kwargs...
    )
    α_new = acs.α
    θ_new = acs.θ
    b = copy(amp.newton_equation.b)
    while θ_new > acs.θ_acc && α_new > 1.0e-10
        Xα = α_new * ams.X
        M = get_manifold(amp)
        retract!(M, ams.p_trial, ams.p, Xα, ams.retraction_method)

        rhs_next = amp.newton_equation(M, get_vectorbundle(amp), ams.p, ams.p_trial)
        rhs_simplified = rhs_next - (1.0 - α_new) * b
        amp.newton_equation.b .= rhs_simplified

        simplified_newton = ams.sub_problem(amp, ams)

        add_arg = (has_components(M) && !ismissing(acs.outer_norm)) ? (outer_norm = acs.outer_norm,) : ()
        nom = norm(amp.manifold, ams.p, simplified_newton, add_arg...)
        denom = norm(amp.manifold, ams.p, ams.X, add_arg...)
        θ_new = nom / denom

        α_new = min(1.0, ((α_new * acs.θ_des) / θ_new))
    end
    amp.newton_equation.b .= b
    acs.last_stepsize = α_new
    return acs.last_stepsize
end
get_initial_stepsize(s::AffineCovariantStepsize) = s.α

function get_last_stepsize(step::AffineCovariantStepsize, ::Any...)
    return step.last_stepsize
end

default_stepsize(M::AbstractManifold, ::Type{VectorBundleNewtonState}) = ConstantStepsize(M)

function status_summary(vbns::VectorBundleNewtonState; context::Symbol = :default)
    (context === :short) && return repr(vbns)
    i = get_count(vbns, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(vbns.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the vector bundle Newton solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(vbns.stop) ? "Yes" : "No"
    _is_inline(context) && (return "$(repr(vbns)) – $(Iter) $(has_converged(vbns) ? "(converged)" : "")")
    s = """
    # Solver state for `Manopt.jl`s Vector bundle Newton method
    $Iter
    ## Parameters
    * retraction method: $(vbns.retraction_method)

    ## Stepsize
    $(_in_str(status_summary(vbns.stepsize; context = context); indent = 0, headers = 1))

    ## Stopping criterion
    $(_in_str(status_summary(vbns.stop; context = context); indent = 0, headers = 1))
    This indicates convergence: $Conv"""
    return s
end


@doc """
    VectorBundleManoptProblem{M<:AbstractManifold,TV<:AbstractManifold,O} <: AbstractManoptProblem{M}

Model a vector bundle problem, that consists of the domain manifold ``$(_math(:Manifold)))`` that is a $(_link(:AbstractManifold)), the range vector bundle ``$(_tex(:Cal, "E"))`` and the Newton equation ``Q_{F(x)}∘ F'(x) δ x + F(x) = 0_{p(F(x))}``.
The Newton equation should be implemented as a functor that computes a representation of the Newton matrix and the right hand side. It needs to have a field ``A`` to store a representation of the Newton matrix ``Q_{F(x)}∘ F'(x) `` and a field ``b`` to store a representation of the right hand side ``F(x)``.
"""
struct VectorBundleManoptProblem{
        M <: AbstractManifold, TV <: AbstractManifold, O,
    } <: AbstractManoptProblem{M}
    manifold::M
    vectorbundle::TV
    newton_equation::O
end

function Base.show(io::IO, vbmp::VectorBundleManoptProblem)
    print(io, "VectorBundleManoptProblem(")
    show(io, vbmp.manifold); print(io, ", ")
    show(io, vbmp.vectorbundle); print(io, ", ")
    show(io, vbmp.newton_equation); print(io, ")")
    return io
end

function status_summary(vbmp::VectorBundleManoptProblem; context::Symbol = :default)
    (context === :short) && return repr(vbmp)
    (context === :inline) && return "A vector bundle problem defined on $(vbmp.manifold) with range $(vbmp.vectorbundle) and newton equation $(vbmp.newton_equation)"
    return """
    A vector bundle problem representing a vector bundle newton equation objective

    ## Manifold
    $(_MANOPT_INDENT)$(replace("$(vbmp.manifold)", "\n#" => "\n$(_MANOPT_INDENT)##", "\n" => "\n$(_MANOPT_INDENT)"))

    ## Range
    $(_MANOPT_INDENT)$(replace("$(vbmp.vectorbundle)", "\n#" => "\n$(_MANOPT_INDENT)##", "\n" => "\n$(_MANOPT_INDENT)"))

    ## Vector bundle newton equation
    $(_MANOPT_INDENT)$(replace("$(vbmp.newton_equation)", "\n#" => "\n$(_MANOPT_INDENT)##", "\n" => "\n$(_MANOPT_INDENT)"))
    """
end

@doc """
    get_vectorbundle(vbp::VectorBundleManoptProblem)

returns the range vector bundle stored within a [`VectorBundleManoptProblem`](@ref)
"""
get_vectorbundle(vbp::VectorBundleManoptProblem) = vbp.vectorbundle

raw"""
    get_manifold(vbp::VectorBundleManoptProblem)

    returns the domain manifold stored within a [`VectorBundleManoptProblem`](@ref)
"""
get_manifold(vbp::VectorBundleManoptProblem) = vbp.manifold

raw"""
    get_newton_equation(mp::VectorBundleManoptProblem)

returns the Newton equation [`newton_equation`](@ref) stored within an [`VectorBundleManoptProblem`](@ref).
"""
function get_newton_equation(vbp::VectorBundleManoptProblem)
    return vbp.newton_equation
end

doc_vector_bundle_newton = """
    vectorbundle_newton(M, E, NE, p; kwargs...)
    vectorbundle_newton!(M, E, NE, p; kwargs...)

Perform Newton's method for finding a zero of a mapping ``F:$(_math(:Manifold))) → $(_tex(:Cal, "E"))`` where ``$(_math(:Manifold)))`` is a manifold and ``$(_tex(:Cal, "E"))`` is a vector bundle.
In each iteration the Newton equation

```math
Q_{F(p)} ∘ F'(p) X + F(p) = 0
```

is solved to compute a Newton direction ``X``.
The next iterate is then computed by applying a retraction.

For more details see [WeiglSchiela:2024, WeiglBergmannSchiela:2025](@cite).

# Arguments

$(_args(:M))
* `E`: range vector bundle
$(_args(:p))
* `NE`: functor representing the Newton equation. It has at least fields ``A`` and ``b`` to store a representation of the Newton matrix ``Q_{F(p)}∘ F'(p)`` (covariant derivative of ``F`` at ``p``) and the right hand side ``F(p)`` at a point ``p ∈ $(_math(:Manifold)))``. The point ``p`` denotes the starting point. The algorithm can be run in-place of ``p``.

# Keyword arguments

$(_kwargs(:sub_problem; default = "nothing")), i.e. you have to provide a method for solving the Newton equation.
  Currently only the closed form solution is implemented, that is, this is a functor that maps either
  `(problem::`[`VectorBundleManoptProblem`](@ref)`, state::VectorBundleNewtonState) -> X` or `(problem, X, state) -> X` to compute the Newton direction.
$(_kwargs(:sub_state; default = "`[`AllocatingEvaluation`](@ref)` "))
$(_kwargs(:retraction_method))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`VectorBundleNewtonState`](@ref)`)"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)"))
$(_kwargs(:X; add_properties = [:as_Memory]))
"""

@doc "$(doc_vector_bundle_newton)"
function vectorbundle_newton(M::AbstractManifold, E::AbstractManifold, NE, p; kwargs...)
    #replace type of E with VectorBundle once this is available in ManifoldsBase
    q = copy(M, p)
    return vectorbundle_newton!(M, E, NE, q; kwargs...)
end


@doc "$(doc_vector_bundle_newton)"
function vectorbundle_newton!(
        M::AbstractManifold, E::AbstractManifold, NE::O, p::P;
        sub_problem::Pr = nothing,
        sub_state::Op = AllocatingEvaluation(),
        X::T = zero_vector(M, p),
        retraction_method::RM = default_retraction_method(M, typeof(p)),
        stopping_criterion::SC = StopAfterIteration(1000),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, VectorBundleNewtonState
        ),
        kwargs...,
    ) where {O, P, T, Pr, Op, RM <: AbstractRetractionMethod, SC <: StoppingCriterion}
    isnothing(sub_problem) && error("Please provide a sub_problem (method that solves the Newton equation)")

    vbp = VectorBundleManoptProblem(M, E, NE)

    vbs = VectorBundleNewtonState(
        M,
        E,
        p,
        sub_problem,
        sub_state;
        X = X,
        retraction_method = retraction_method,
        stopping_criterion = stopping_criterion,
        stepsize = _produce_type(stepsize, M, p)
    )
    dvbs = decorate_state!(vbs; kwargs...)
    solve!(vbp, dvbs)
    return get_solver_return(dvbs)
end

function initialize_solver!(::VectorBundleManoptProblem, s::VectorBundleNewtonState)
    return s
end

# TODO: When needed: add the variant of iterative solvers for the Newton equation's sub problem

# Closed form solution of the sub-problem, allocating variant
function step_solver!(
        mp::VectorBundleManoptProblem,
        s::VectorBundleNewtonState{P, T, PR, AllocatingEvaluation},
        k,
    ) where {P, T, PR}
    M = get_manifold(mp) # domain manifold
    E = get_vectorbundle(mp) # vector bundle (codomain of F)
    # update Newton matrix and right hand side
    mp.newton_equation(M, E, s.p)
    # compute Newton direction
    s.X = s.sub_problem(mp, s)
    #compute a stepsize
    step = s.stepsize(mp, s, k)
    # retract
    ManifoldsBase.retract_fused!(get_manifold(mp), s.p, s.p, s.X, step, s.retraction_method)
    s.p_trial = copy(get_manifold(mp), s.p) # needed for affine covariant damping (can be ignored if this stepsize computation is not used)
    return s
end

# Closed form solution of the sub-problem, in-place variant
function step_solver!(
        mp::VectorBundleManoptProblem, s::VectorBundleNewtonState{P, T, PR, InplaceEvaluation}, k
    ) where {P, T, PR}
    M = get_manifold(mp) # domain manifold
    E = get_vectorbundle(mp) # vector bundle (codomain of F)
    # update Newton matrix and right hand side
    mp.newton_equation(M, E, s.p)
    # compute Newton direction (in-place)
    s.sub_problem(mp, s.X, s)
    step = s.stepsize(mp, s, k)
    # retract
    ManifoldsBase.retract_fused!(M, s.p, s.p, s.X, step, s.retraction_method)
    s.p_trial = copy(M, s.p) # needed for affine covariant damping (can be ignored if this stepsize computation is not used)
    return s
end
