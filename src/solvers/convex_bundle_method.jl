@doc """
    estimate_sectional_curvature(M::AbstractManifold, p)

Estimate the sectional curvature of a manifold ``$(_math(:Manifold)))`` at a point ``p ∈ $(_math(:Manifold)))``
on two random tangent vectors at ``p`` that are orthogonal to each other.

# See also

[`sectional_curvature`](@extref `ManifoldsBase.sectional_curvature-Tuple{AbstractManifold, Any, Any, Any}`)
"""
function estimate_sectional_curvature(M::AbstractManifold, p)
    X = rand(M; vector_at = p)
    Y = rand(M; vector_at = p)
    Y = Y - (inner(M, p, X, Y) / norm(M, p, X)^2 * X)
    return sectional_curvature(M, p, X, Y)
end

@doc """
    ζ_1(ω, δ)

compute a curvature-dependent bound.
The formula reads

```math
ζ_{1, ω}(δ)
:=
$(
    _tex(
        :cases,
        "1 & $(_tex(:text, " if ")) ω ≥ 0,",
        "$(_tex(:sqrt, "-ω"))δ$(_tex(:cot))($(_tex(:sqrt, "-ω"))δ) & $(_tex(:text, " if ")) ω < 0",
    )
)
```

where ``ω ≤ κ_p`` for all ``p ∈ $(_tex(:Cal, "U"))`` is a lower bound to the sectional curvature in
a (strongly geodesically convex) bounded subset ``$(_tex(:Cal, "U")) ⊆ $(_math(:Manifold)))`` with diameter ``δ``.
"""
function ζ_1(k_min, diameter)
    (k_min < zero(k_min)) && return sqrt(-k_min) * diameter * coth(sqrt(-k_min) * diameter)
    return one(k_min)
end

@doc """
    ζ_2(Ω, δ)

compute a curvature-dependent bound.
The formula reads

```math
ζ_{2, Ω}(δ) :=
$(
    _tex(
        :cases,
        "1 & $(_tex(:text, " if ")) Ω ≤ 0,",
        "$(_tex(:sqrt, "Ω"))δ$(_tex(:cot))($(_tex(:sqrt, "Ω"))δ) & $(_tex(:text, " if ")) Ω > 0",
    )
)
```

where ``Ω ≥ κ_p`` for all ``p ∈ $(_tex(:Cal, "U"))`` is an upper bound to the sectional curvature in
a (strongly geodesically convex) bounded subset ``$(_tex(:Cal, "U")) ⊆ $(_math(:Manifold)))`` with diameter ``δ``.
"""
function ζ_2(k_max, diameter)
    (k_max > zero(k_max)) && return sqrt(k_max) * diameter * cot(sqrt(k_max) * diameter)
    return one(k_max)
end

@doc """
    close_point(M, p, tol; retraction_method=default_retraction_method(M, typeof(p)))

sample a random point close to ``p ∈ $(_math(:Manifold)))`` within a tolerance `tol`
and a [retraction](@extref ManifoldsBase :doc:`retractions`).
"""
function close_point(M, p, tol; retraction_method = default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at = p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end

@doc """
    ConvexBundleMethodState <: AbstractManoptSolverState

Stores option values for a [`convex_bundle_method`](@ref) solver.

# Fields

THe following fields require a (real) number type `R`, as well as
point type `P` and a tangent vector type `T``

* `atol_λ::R`:                 tolerance parameter for the convex coefficients in λ
* `atol_errors::R:             tolerance parameter for the linearization errors
* `bundle<:AbstractVector{Tuple{<:P,<:T}}`: bundle that collects each iterate with the computed subgradient at the iterate
* `bundle_cap::Int`: the maximal number of elements the bundle is allowed to remember
* `diameter::R`: estimate for the diameter of the level set of the objective function at the starting point
* `domain: the domain of ``f`` as a function `(M,p) -> b`that evaluates to true when the current candidate is in the domain of `f`, and false otherwise,
* `g::T`:                      descent direction
$(_fields(:inverse_retraction_method))
* `k_max::R`:                  upper bound on the sectional curvature of the manifold
* `linearization_errors<:AbstractVector{<:R}`: linearization errors at the last serious step
* `m::R`:                      the parameter to test the decrease of the cost: ``f(q_{k+1}) ≤ f(p_k) + m ξ``.
$(_fields(:p; add_properties = [:as_Iterate]))
* `p_last_serious::P`:         last serious iterate
$(_fields(:retraction_method))
$(_fields(:stopping_criterion; name = "stop"))
* `transported_subgradients`:  subgradients of the bundle that are transported to `p_last_serious`
$(_fields(:vector_transport_method))
$(_fields(:X; add_properties = [:as_Subgradient]))
$(_fields(:stepsize))
* `ε::R`:                      convex combination of the linearization errors
* `λ:::AbstractVector{<:R}`:   convex coefficients from the slution of the subproblem
* `ξ`:                         the stopping parameter given by ``ξ = -$(_tex(:norm, "g"))^2 – ε``
$(_fields([:sub_problem, :sub_state]))

# Constructor

    ConvexBundleMethodState(M::AbstractManifold, sub_problem, sub_state; kwargs...)
    ConvexBundleMethodState(M::AbstractManifold, sub_problem=convex_bundle_method_subsolver; evaluation=AllocatingEvaluation(), kwargs...)

Generate the state for the [`convex_bundle_method`](@ref) on the manifold `M`

## Input

$(_args([:M, :sub_problem, :sub_state]))

# Keyword arguments

Most of the following keyword arguments set default values for the fields mentioned before.

* `atol_λ=eps()`
* `atol_errors=eps()`
* `bundle_cap=25``
* `m=1e-2`
* `diameter=50.0`
* `domain=(M, p) -> isfinite(f(M, p))`
* `k_max=0`
* `k_min=0`
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`ConvexBundleMethodState`](@ref)`)"))
$(_kwargs([:inverse_retraction_method, :retraction_method]))
$(_kwargs(:stopping_criterion; default = "`[`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`$(_sc(:Any))[`StopAfterIteration`](@ref)`(5000)"))
$(_kwargs(:X))
  to specify the type of tangent vector to use.
$(_kwargs(:vector_transport_method))
"""
mutable struct ConvexBundleMethodState{
        P,
        T,
        Pr <: Union{F, AbstractManoptProblem} where {F},
        St <: AbstractManoptSolverState,
        R,
        A <: AbstractVector{<:R},
        B <: AbstractVector{Tuple{<:P, <:T}},
        C <: AbstractVector{T},
        D,
        I,
        IR <: AbstractInverseRetractionMethod,
        TR <: AbstractRetractionMethod,
        TS <: Stepsize,
        TSC <: StoppingCriterion,
        VT <: AbstractVectorTransportMethod,
    } <: AbstractManoptSolverState where {R <: Real, P, T, I <: Int, Pr}
    atol_λ::R
    atol_errors::R
    bundle::B
    bundle_cap::I
    diameter::R
    domain::D
    g::T
    inverse_retraction_method::IR
    k_max::R
    k_min::R
    last_stepsize::R
    null_stepsize::R
    linearization_errors::A
    m::R
    p::P
    p_last_serious::P
    retraction_method::TR
    stepsize::TS
    stop::TSC
    transported_subgradients::C
    vector_transport_method::VT
    X::T
    ε::R
    ξ::R
    λ::A
    sub_problem::Pr
    sub_state::St
    ϱ::R
    function ConvexBundleMethodState(
            M::TM,
            sub_problem::Pr,
            sub_state::St;
            p::P = rand(M),
            p_estimate = p,
            atol_λ::R = eps(),
            atol_errors::R = eps(),
            bundle_cap::I = 25,
            m::R = 1.0e-2,
            diameter::R = 50.0,
            domain::D = (M, p) -> isfinite(f(M, p)),
            k_max = nothing,
            k_min = nothing,
            k_size = 100,
            last_stepsize = one(number_eltype(atol_λ)),
            stepsize::S = default_stepsize(M, ConvexBundleMethodState),
            inverse_retraction_method::IR = default_inverse_retraction_method(M, typeof(p)),
            retraction_method::TR = default_retraction_method(M, typeof(p)),
            stopping_criterion::SC = StopWhenLagrangeMultiplierLess(1.0e-8) |
                StopAfterIteration(5000),
            X::T = zero_vector(M, p),
            vector_transport_method::VT = default_vector_transport_method(M, typeof(p)),
            ϱ = nothing,
        ) where {
            D,
            IR <: AbstractInverseRetractionMethod,
            P,
            T,
            Pr <: Union{AbstractManoptProblem, F} where {F},
            St <: AbstractManoptSolverState,
            I,
            TM <: AbstractManifold,
            TR <: AbstractRetractionMethod,
            SC <: StoppingCriterion,
            S <: Stepsize,
            VT <: AbstractVectorTransportMethod,
            R <: Real,
        }
        bundle = Vector{Tuple{P, T}}()
        g = zero_vector(M, p)
        null_stepsize = one(R)
        linearization_errors = Vector{R}()
        transported_subgradients = Vector{T}()
        ε = zero(R)
        λ = Vector{R}()
        ξ = zero(R)
        if ϱ === nothing
            if (k_max === nothing)
                estimation_points = [
                    close_point(
                            M, p_estimate, diameter / 3; retraction_method = retraction_method
                        ) for _ in 1:k_size
                ]
                estimation_vectors_1 = [rand(M; vector_at = pe) for pe in estimation_points]
                estimation_vectors_2 = [rand(M; vector_at = pe) for pe in estimation_points]
                s = [
                    sectional_curvature(
                            M,
                            estimation_points[i],
                            estimation_vectors_1[i],
                            estimation_vectors_2[i],
                        ) for i in 1:k_size
                ]
            end
            (k_min === nothing) && (k_min = minimum(s))
            (k_max === nothing) && (k_max = maximum(s))
            ϱ = max(ζ_1(k_min, diameter) - one(k_min), one(k_max) - ζ_2(k_max, diameter))
        end
        return new{
            P,
            T,
            Pr,
            St,
            R,
            typeof(linearization_errors),
            typeof(bundle),
            typeof(transported_subgradients),
            D,
            I,
            IR,
            TR,
            S,
            SC,
            VT,
        }(
            atol_λ,
            atol_errors,
            bundle,
            bundle_cap,
            diameter,
            domain,
            g,
            inverse_retraction_method,
            k_max,
            k_min,
            last_stepsize,
            null_stepsize,
            linearization_errors,
            m,
            p,
            copy(M, p),
            retraction_method,
            stepsize,
            stopping_criterion,
            transported_subgradients,
            vector_transport_method,
            X,
            ε,
            ξ,
            λ,
            sub_problem,
            sub_state,
            ϱ,
        )
    end
end
function ConvexBundleMethodState(
        M::AbstractManifold,
        sub_problem = convex_bundle_method_subsolver;
        evaluation::E = AllocatingEvaluation(),
        kwargs...,
    ) where {E <: AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return ConvexBundleMethodState(M, sub_problem, cfs; kwargs...)
end

get_iterate(bms::ConvexBundleMethodState) = bms.p_last_serious
function set_iterate!(bms::ConvexBundleMethodState, M, p)
    copyto!(M, bms.p_last_serious, p)
    return bms
end
get_subgradient(bms::ConvexBundleMethodState) = bms.g
function default_stepsize(M::AbstractManifold, ::Type{ConvexBundleMethodState})
    return ConstantStepsize(M)
end
function show(io::IO, cbms::ConvexBundleMethodState)
    i = get_count(cbms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(cbms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Convex Bundle Method
    $Iter
    ## Parameters
    * tolerance parameter for the convex coefficients:  $(cbms.atol_λ)
    * tolerance parameter for the linearization errors: $(cbms.atol_errors)
    * bundle cap size:                                  $(cbms.bundle_cap)
    * current bundle size:                              $(length(cbms.bundle))
    * curvature upper bound:                            $(cbms.k_max)
    * descent test parameter:                           $(cbms.m)
    * diameter:                                         $(cbms.diameter)
    * inverse retraction:                               $(cbms.inverse_retraction_method)
    * retraction:                                       $(cbms.retraction_method)
    * Lagrange parameter value:                         $(cbms.ξ)
    * vector transport:                                 $(cbms.vector_transport_method)

    ## Stopping criterion
    $(status_summary(cbms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

function _domain_condition(M, q, p, t, length, domain)
    return (!domain(M, q) || (distance(M, p, q) < t * length))
end

function _null_condition(amp, M, q, p_last_serious, X, g, VT, IRT, m, t, ξ, ϱ)
    return (
        inner(M, p_last_serious, vector_transport_to(M, q, X, p_last_serious, VT), t * g) ≥
            -m * t * ξ - (
            get_cost(amp, p_last_serious) - get_cost(amp, q) -
                inner(M, q, X, inverse_retract(M, q, p_last_serious, IRT)) -
                ϱ * norm(M, q, X) * norm(M, q, inverse_retract(M, q, p_last_serious, IRT))
        )
    )
end

@doc """
    DomainBackTrackingStepsize <: Stepsize

Implement a backtrack as long as we are ``q =$(_tex(:retr))_p(X)``
yields a point closer to ``p`` than ``$(_tex(:norm, "X"; index = "p"))`` or
``q`` is not on the domain.
For the domain this step size requires a [`ConvexBundleMethodState`](@ref).
"""
mutable struct DomainBackTrackingStepsize{TRM <: AbstractRetractionMethod, P, F} <: Stepsize
    candidate_point::P
    contraction_factor::F
    initial_stepsize::F
    last_stepsize::F
    message::String
    retraction_method::TRM
    function DomainBackTrackingStepsize(
            M::AbstractManifold;
            candidate_point::P = allocate_result(M, rand),
            contraction_factor::F = 0.95,
            initial_stepsize::F = 1.0,
            retraction_method::TRM = default_retraction_method(M),
        ) where {TRM, P, F}
        return new{TRM, P, F}(
            candidate_point,
            contraction_factor,
            initial_stepsize,
            initial_stepsize,
            "", # initialize an empty message
            retraction_method,
        )
    end
end
function (dbt::DomainBackTrackingStepsize)(
        amp::AbstractManoptProblem, cbms::ConvexBundleMethodState, ::Int; kwargs...
    )
    M = get_manifold(amp)
    dbt.last_stepsize = 1.0
    retract!(
        M,
        dbt.candidate_point,
        cbms.p_last_serious,
        -dbt.last_stepsize * cbms.g,
        dbt.retraction_method,
    )
    while _domain_condition(
            M,
            dbt.candidate_point,
            cbms.p_last_serious,
            dbt.last_stepsize,
            norm(M, cbms.p_last_serious, cbms.g),
            cbms.domain,
        )
        dbt.last_stepsize *= dbt.contraction_factor
        retract!(
            M,
            dbt.candidate_point,
            cbms.p_last_serious,
            -dbt.last_stepsize * cbms.g,
            dbt.retraction_method,
        )
    end
    return dbt.last_stepsize
end
get_initial_stepsize(dbt::DomainBackTrackingStepsize) = dbt.initial_stepsize
function show(io::IO, dbt::DomainBackTrackingStepsize)
    return print(
        io,
        """
        DomainBackTracking(;
            initial_stepsize=$(dbt.initial_stepsize)
            retraction_method=$(dbt.retraction_method)
            contraction_factor=$(dbt.contraction_factor)
        )""",
    )
end
function status_summary(dbt::DomainBackTrackingStepsize)
    return "$(dbt)\nand a computed last stepsize of $(dbt.last_stepsize)"
end
get_message(dbt::DomainBackTrackingStepsize) = dbt.message
function get_parameter(dbt::DomainBackTrackingStepsize, s::Val{:Iterate})
    return dbt.candidate_point
end

"""
    DomainBackTracking(; kwargs...)
    DomainBackTracking(M::AbstractManifold; kwargs...)

Specify a step size that performs a backtracking to the interior of the domain of the objective function.

# Keyword arguments

* `candidate_point=allocate_result(M, rand)`:
  specify a point to be used as memory for the candidate points.
* `contraction_factor`: how to update ``s`` in the decrease step
* `initial_stepsize``: specify an initial step size
$(_kwargs(:retraction_method))

$(_note(:ManifoldDefaultFactory, "DomainBackTrackingStepsize"))
"""
function DomainBackTracking(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.DomainBackTrackingStepsize, args...; kwargs...)
end

@doc """
    NullStepBackTrackingStepsize <: Stepsize

Implement a backtracking with a geometric condition in the case of a null step.
For the domain this step size requires a [`ConvexBundleMethodState`](@ref).
"""
mutable struct NullStepBackTrackingStepsize{TRM <: AbstractRetractionMethod, P, F, T} <: Stepsize
    candidate_point::P
    contraction_factor::F
    initial_stepsize::F
    last_stepsize::F
    message::String
    retraction_method::TRM
    X::T
    function NullStepBackTrackingStepsize(
            M::AbstractManifold;
            candidate_point::P = allocate_result(M, rand),
            contraction_factor::F = 0.95,
            initial_stepsize::F = 1.0,
            retraction_method::TRM = default_retraction_method(M),
            X::T = zero_vector(M, candidate_point),
        ) where {TRM, P, F, T}
        return new{TRM, P, F, T}(
            candidate_point,
            contraction_factor,
            initial_stepsize,
            initial_stepsize,
            "", # initialize an empty message
            retraction_method,
            X,
        )
    end
end
function (nsbt::NullStepBackTrackingStepsize)(
        amp::AbstractManoptProblem, cbms::ConvexBundleMethodState, ::Int, kwargs...
    )
    M = get_manifold(amp)
    nsbt.last_stepsize = cbms.last_stepsize
    retract!(
        M,
        nsbt.candidate_point,
        cbms.p_last_serious,
        -nsbt.last_stepsize * cbms.g,
        nsbt.retraction_method,
    )
    get_subgradient!(amp, nsbt.X, nsbt.candidate_point)
    while _null_condition(
            amp,
            M,
            nsbt.candidate_point,
            cbms.p_last_serious,
            nsbt.X,
            cbms.g,
            cbms.vector_transport_method,
            cbms.inverse_retraction_method,
            cbms.m,
            nsbt.last_stepsize,
            cbms.ξ,
            cbms.ϱ,
        )
        nsbt.last_stepsize *= nsbt.contraction_factor
        retract!(
            M,
            nsbt.candidate_point,
            cbms.p_last_serious,
            -nsbt.last_stepsize * cbms.g,
            nsbt.retraction_method,
        )
        get_subgradient!(amp, nsbt.X, nsbt.candidate_point)
    end
    return nsbt.last_stepsize
end
get_initial_stepsize(nsbt::NullStepBackTrackingStepsize) = nsbt.initial_stepsize
function get_parameter(nsbt::NullStepBackTrackingStepsize, s::Val{:Iterate})
    return nsbt.candidate_point
end
function get_parameter(nsbt::NullStepBackTrackingStepsize, s::Val{:Subgradient})
    return nsbt.X
end
function show(io::IO, nsbt::NullStepBackTrackingStepsize)
    return print(
        io,
        """
        NullStepBackTracking(;
            initial_stepsize=$(nsbt.initial_stepsize)
            retraction_method=$(nsbt.retraction_method)
            contraction_factor=$(nsbt.contraction_factor)
            candidate_point=$(nsbt.candidate_point)
            X=$(nsbt.X)
            last_stepsize=$(nsbt.last_stepsize)
        )""",
    )
end
function status_summary(nsbt::NullStepBackTrackingStepsize)
    return "$(nsbt)\nand a computed last stepsize of $(nsbt.last_stepsize)"
end
get_message(nsbt::NullStepBackTrackingStepsize) = nsbt.message

_doc_cbm_gk = """
```math
g_k = $(_tex(:sum, "j ∈ J_k")) λ_j^k $(_tex(:rm, "P"))_{p_k←q_j}X_{q_j},
```
"""
_doc_convex_bundle_method = """
    convex_bundle_method(M, f, ∂f, p)
    convex_bundle_method!(M, f, ∂f, p)

perform a convex bundle method ``p^{(k+1)} = $(_tex(:retr))_{p^{(k)}}(-g_k)`` where

$(_doc_cbm_gk)

and ``p_k`` is the last serious iterate, ``X_{q_j} ∈ ∂f(q_j)``, and the ``λ_j^k`` are solutions
to the quadratic subproblem provided by the [`convex_bundle_method_subsolver`](@ref).

Though the subdifferential might be set valued, the argument `∂f` should always
return one element from the subdifferential, but not necessarily deterministic.

For more details, see [BergmannHerzogJasa:2024](@cite).

# Input

$(_args([:M, :f, :subgrad_f, :p]))

# Keyword arguments

* `atol_λ=eps()` : tolerance parameter for the convex coefficients in ``λ``.
* `atol_errors=eps()`: : tolerance parameter for the linearization errors.
* `bundle_cap=25``
* `m=1e-3`: : the parameter to test the decrease of the cost: ``f(q_{k+1}) ≤ f(p_k) + m ξ``.
* `diameter=50.0`: estimate for the diameter of the level set of the objective function at the starting point.
* `domain=(M, p) -> isfinite(f(M, p))`: a function to that evaluates to true when the current candidate is in the domain of the objective `f`, and false otherwise.
$(_kwargs(:evaluation))
* `k_max=0`: upper bound on the sectional curvature of the manifold.
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`ConvexBundleMethodState`](@ref)`)"))
$(_kwargs(:inverse_retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopWhenLagrangeMultiplierLess`](@ref)`(1e-8)`$(_sc(:Any))[`StopAfterIteration`](@ref)`(5000)"))
$(_kwargs(:vector_transport_method))
$(_kwargs(:sub_state; default = "`[`convex_bundle_method_subsolver`](@ref)"))
$(_kwargs(:sub_problem; default = "`[`AllocatingEvaluation`](@ref)´ "))
$(_kwargs(:X))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_convex_bundle_method)"
function convex_bundle_method(
        M::AbstractManifold, f::TF, ∂f::TdF, p = rand(M); kwargs...
    ) where {TF, TdF}
    keywords_accepted(convex_bundle_method; kwargs...)
    p_star = copy(M, p)
    return convex_bundle_method!(M, f, ∂f, p_star; kwargs...)
end
calls_with_kwargs(::typeof(convex_bundle_method)) = (convex_bundle_method!,)

@doc "$(_doc_convex_bundle_method)"
function convex_bundle_method!(
        M::AbstractManifold,
        f::TF,
        ∂f!!::TdF,
        p;
        atol_λ::R = sqrt(eps()),
        atol_errors::R = sqrt(eps()),
        bundle_cap::Int = 25,
        contraction_factor = 0.975,
        diameter::R = π / 3, # was `k_max -> k_max === nothing ? π/2 : (k_max ≤ zero(R) ? typemax(R) : π/3)`,
        domain = (M, p) -> isfinite(f(M, p)),
        m::R = 1.0e-3,
        k_max = 0,
        k_min = 0,
        p_estimate = p,
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = DomainBackTracking(;
            contraction_factor = contraction_factor
        ),
        debug = [DebugWarnIfLagrangeMultiplierIncreases()],
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        inverse_retraction_method::IR = default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TRetr = default_retraction_method(M, typeof(p)),
        stopping_criterion::StoppingCriterion = StopWhenAny(
            StopWhenLagrangeMultiplierLess(1.0e-8; names = ["-ξ"]), StopAfterIteration(5000)
        ),
        vector_transport_method::VTransp = default_vector_transport_method(M, typeof(p)),
        sub_problem = convex_bundle_method_subsolver,
        sub_state::Union{AbstractEvaluationType, AbstractManoptSolverState} = evaluation,
        ϱ = nothing,
        kwargs...,
    ) where {R <: Real, TF, TdF, TRetr, IR, VTransp}
    keywords_accepted(convex_bundle_method!; kwargs...)
    sgo = ManifoldSubgradientObjective(f, ∂f!!; evaluation = evaluation)
    dsgo = decorate_objective!(M, sgo; kwargs...)
    mp = DefaultManoptProblem(M, dsgo)
    sub_state_storage = maybe_wrap_evaluation_type(sub_state)
    bms = ConvexBundleMethodState(
        M,
        sub_problem,
        maybe_wrap_evaluation_type(sub_state);
        p = p,
        atol_λ = atol_λ,
        atol_errors = atol_errors,
        bundle_cap = bundle_cap,
        diameter = diameter,
        domain = domain,
        m = m,
        k_max = k_max,
        k_min = k_min,
        p_estimate = p_estimate,
        stepsize = _produce_type(stepsize, M),
        inverse_retraction_method = inverse_retraction_method,
        retraction_method = retraction_method,
        stopping_criterion = stopping_criterion,
        vector_transport_method = vector_transport_method,
        ϱ = ϱ,
    )
    bms = decorate_state!(bms; debug = debug, kwargs...)
    return get_solver_return(solve!(mp, bms))
end
calls_with_kwargs(::typeof(convex_bundle_method!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(
        mp::AbstractManoptProblem, bms::ConvexBundleMethodState{P, T, Pr, St, R}
    ) where {P, T, Pr, St, R}
    M = get_manifold(mp)
    copyto!(M, bms.p_last_serious, bms.p)
    get_subgradient!(mp, bms.X, bms.p)
    copyto!(M, bms.g, bms.p_last_serious, bms.X)
    empty!(bms.bundle)
    push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
    empty!(bms.λ)
    push!(bms.λ, zero(R))
    empty!(bms.linearization_errors)
    push!(bms.linearization_errors, zero(R))
    empty!(bms.transported_subgradients)
    push!(bms.transported_subgradients, zero_vector(M, bms.p))
    return bms
end
function step_solver!(mp::AbstractManoptProblem, bms::ConvexBundleMethodState, k)
    M = get_manifold(mp)
    # Refactor to in-place
    for (j, (qj, Xj)) in enumerate(bms.bundle)
        vector_transport_to!(
            M,
            bms.transported_subgradients[j],
            qj,
            Xj,
            bms.p_last_serious,
            bms.vector_transport_method,
        )
    end
    _convex_bundle_subsolver!(M, bms)
    bms.g .= sum(bms.λ .* bms.transported_subgradients)
    bms.ε = sum(bms.λ .* bms.linearization_errors)
    bms.ξ = (-norm(M, bms.p_last_serious, bms.g)^2) - (bms.ε)
    bms.last_stepsize = get_stepsize(mp, bms, k)
    copyto!(M, bms.p, get_parameter(bms.stepsize, :Iterate))
    if get_cost(mp, bms.p) ≤
            (get_cost(mp, bms.p_last_serious) + bms.last_stepsize * bms.m * bms.ξ)
        copyto!(M, bms.p_last_serious, bms.p)
        get_subgradient!(mp, bms.X, bms.p)
    else
        # Condition for null-steps
        nsbt = NullStepBackTrackingStepsize(
            M;
            contraction_factor = bms.stepsize.contraction_factor,
            initial_stepsize = bms.last_stepsize,
        )
        bms.null_stepsize = nsbt(mp, bms, k)
        copyto!(M, bms.p, get_parameter(nsbt, :Iterate))
        copyto!(M, bms.X, get_parameter(nsbt, :Subgradient))
    end
    v = findall(λj -> λj ≤ bms.atol_λ, bms.λ)
    if !isempty(v)
        deleteat!(bms.bundle, v)
        # Update sizes of subgradient and lambda linearization errors as well
        deleteat!(bms.λ, v)
        deleteat!(bms.linearization_errors, v)
        deleteat!(bms.transported_subgradients, v)
    end
    l = length(bms.bundle)
    if l == bms.bundle_cap && bms.bundle[1][1] ≠ bms.p_last_serious
        #
        deleteat!(bms.bundle, 1)
        deleteat!(bms.λ, 1)
        deleteat!(bms.linearization_errors, 1)
        push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
        push!(bms.linearization_errors, 0.0)
        push!(bms.λ, 0.0)
    else
        # push to bundle and update subgradients, λ, and linearization_errors (+1 in length)
        push!(bms.bundle, (copy(M, bms.p), copy(M, bms.p, bms.X)))
        push!(bms.linearization_errors, 0.0)
        push!(bms.λ, 0.0)
        push!(bms.transported_subgradients, zero_vector(M, bms.p))
    end
    for (j, (qj, Xj)) in enumerate(bms.bundle)
        bms.linearization_errors[j] =
            get_cost(mp, bms.p_last_serious) - get_cost(mp, qj) - (
            inner(
                M,
                qj,
                Xj,
                inverse_retract(M, qj, bms.p_last_serious, bms.inverse_retraction_method),
            )
        ) + (
            bms.ϱ *
                norm(M, qj, Xj) *
                norm(
                M,
                qj,
                inverse_retract(
                    M, qj, bms.p_last_serious, bms.inverse_retraction_method
                ),
            )
        )
    end
    return bms
end
get_solver_result(bms::ConvexBundleMethodState) = bms.p_last_serious
function get_last_stepsize(::AbstractManoptProblem, bms::ConvexBundleMethodState, k)
    return bms.last_stepsize
end

#
#
# Dispatching on different types of sub solvers
# (a) closed form allocating
function _convex_bundle_subsolver!(
        M, bms::ConvexBundleMethodState{P, T, F, ClosedFormSubSolverState{AllocatingEvaluation}}
    ) where {P, T, F}
    bms.λ = bms.sub_problem(
        M, bms.p_last_serious, bms.linearization_errors, bms.transported_subgradients
    )
    return bms
end
# (b) closed form in-place
function _convex_bundle_subsolver!(
        M, bms::ConvexBundleMethodState{P, T, F, ClosedFormSubSolverState{InplaceEvaluation}}
    ) where {P, T, F}
    bms.sub_problem(
        M, bms.λ, bms.p_last_serious, bms.linearization_errors, bms.transported_subgradients
    )
    return bms
end
# (c) TODO: implement the case where problem and state are given and `solve!` is called

#
# Lagrange stopping criterion
function (sc::StopWhenLagrangeMultiplierLess)(
        mp::AbstractManoptProblem, bms::ConvexBundleMethodState, k::Int
    )
    if k == 0 # reset on init
        sc.at_iteration = -1
    end
    M = get_manifold(mp)
    if (sc.mode == :estimate) && (-bms.ξ ≤ sc.tolerances[1]) && (k > 0)
        sc.values[1] = -bms.ξ
        sc.at_iteration = k
        return true
    end
    ng = norm(M, bms.p_last_serious, bms.g)
    if (sc.mode == :both) &&
            (bms.ε ≤ sc.tolerances[1]) &&
            (ng ≤ sc.tolerances[2]) &&
            (k > 0)
        sc.values[1] = bms.ε
        sc.values[2] = ng
        sc.at_iteration = k
        return true
    end
    return false
end
function (d::DebugWarnIfLagrangeMultiplierIncreases)(
        ::AbstractManoptProblem, st::ConvexBundleMethodState, k::Int
    )
    (k < 1) && (return nothing)
    if d.status !== :No
        new_value = -st.ξ
        if new_value ≥ d.old_value * d.tol
            @warn """The Lagrange multiplier increased by at least $(d.tol).
            At iteration #$k the negative of the Lagrange multiplier, -ξ, increased from $(d.old_value) to $(new_value).

            Consider decreasing either the `diameter` keyword argument, or one
            of the parameters involved in the estimation of the sectional curvature, such as
            `k_min`, `k_max`, `diameter`, or `ϱ` in the `convex_bundle_method` call.
            of the parameters involved in the estimation of the sectional curvature, such as `k_min`, `k_max`, `diameter`, or `ϱ` in the `convex_bundle_method` call.
            """
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfLagrangeMultiplierIncreases(:Always) to get all warnings."
                d.status = :No
            end
        elseif new_value < zero(number_eltype(st.ξ))
            @warn """The Lagrange multiplier is positive.
            At iteration #$k the negative of the Lagrange multiplier, -ξ, became negative.

            Consider increasing either the `diameter` keyword argument, or changing
            one of the parameters involved in the estimation of the sectional curvature, such as
            `k_min`, `k_max`, `diameter`, or `ϱ` in the `convex_bundle_method` call.
            one of the parameters involved in the estimation of the sectional curvature, such as `k_min`, `k_max`, `diameter`, or `ϱ` in the `convex_bundle_method` call.
            """
        else
            d.old_value = min(d.old_value, new_value)
        end
    end
    return nothing
end

function (d::DebugStepsize)(
        dmp::P, bms::ConvexBundleMethodState, k::Int
    ) where {P <: AbstractManoptProblem}
    (k < 1) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_last_stepsize(dmp, bms, k))
    return nothing
end
