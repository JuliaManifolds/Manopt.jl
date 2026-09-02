"""
    IdentityUpdateRule <: DirectionUpdateRule

The default gradient direction update is the identity, usually it just evaluates the gradient.

You can also use `Gradient()` to create the corresponding factory, though this only delays
this parameter-free instantiation to later.
"""
struct IdentityUpdateRule <: DirectionUpdateRule end
Gradient() = ManifoldDefaultsFactory(Manopt.IdentityUpdateRule; requires_manifold = false)
Base.show(io::IO, agr::IdentityUpdateRule) = print(io, "IdentityUpdateRule()")
function status_summary(ir::IdentityUpdateRule; context::Symbol = :default)
    (context === :short) && return repr(ir)
    return "A gradient processor that evaluates the gradient"
end

"""
    MomentumGradientRule <: DirectionUpdateRule

Store the necessary information to compute the [`MomentumGradient`](@ref)
direction update, see [RoyMhammediHarandi:2018, LeggioScuppa:2026](@cite).

# Fields

$(_fields(:p; name = "p_old"))
* `momentum::Real`: factor for the momentum
* `direction`: internal [`DirectionUpdateRule`](@ref) to determine directions
  to add the momentum to.
$(_fields(:vector_transport_method))
$(_fields(:X; name = "X_old"))

# Constructors

    MomentumGradientRule(M::AbstractManifold; kwargs...)
    MomentumGradientRule(M::AbstractManifold, p; kwargs...)

Initialize a momentum gradient rule to `s`, where `p` and `X` are memory for interim values.

## Keyword arguments

$(_kwargs(:p))
* `s=`[`IdentityUpdateRule`](@ref)`()`
* `momentum=0.2`
$(_kwargs([:vector_transport_method, :X]))

# See also
[`MomentumGradient`](@ref)
"""
mutable struct MomentumGradientRule{
        P, T, D <: DirectionUpdateRule, R <: Real, VTM <: AbstractVectorTransportMethod,
    } <: DirectionUpdateRule
    momentum::R
    p_old::P
    direction::D
    vector_transport_method::VTM
    X_old::T
    function MomentumGradientRule(;
            momentum::R, p_old::P, direction::D, vector_transport_method::VTM, X_old::T
        ) where {P, T, D <: DirectionUpdateRule, R <: Real, VTM <: AbstractVectorTransportMethod}
        return new{P, T, D, R, VTM}(momentum, p_old, direction, vector_transport_method, X_old)
    end
end
function MomentumGradientRule(M::AbstractManifold, p; kwargs...)
    return MomentumGradientRule(M; p = copy(M, p), kwargs...)
end
function MomentumGradientRule(
        M::AbstractManifold;
        p::P = rand(M),
        direction::Union{<:DirectionUpdateRule, ManifoldDefaultsFactory} = Gradient(),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
        X::Q = zero_vector(M, p),
        momentum::F = 0.2,
    ) where {P, Q, F <: Real, VTM <: AbstractVectorTransportMethod}
    dir = _produce_type(direction, M)
    return MomentumGradientRule(;
        momentum = momentum, p_old = p, direction = dir, vector_transport_method = vector_transport_method, X_old = X,
    )
end
function (mg::MomentumGradientRule)(
        mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    step, dir = mg.direction(mp, s, k) #get inner direction and step size
    # store the direction without the step size folded in, so that the solver applies
    # the step exactly once: the displacement is `-step * X_old`
    mg.X_old =
        mg.momentum *
        vector_transport_to(M, mg.p_old, mg.X_old, p, mg.vector_transport_method) + dir
    copyto!(M, mg.p_old, p)
    # return a copy: the solver binds this to its gradient buffer and overwrites it next step
    return step, copy(M, p, mg.X_old)
end
function Base.show(io::IO, mgr::MomentumGradientRule)
    print(io, "MomentumGradientRule(; momentum = ", mgr.momentum)
    print(io, ", p_old = ", mgr.p_old, ", X_old ", mgr.X_old)
    print(io, ", direction = ", mgr.direction)
    print(io, "vector_transport_method = ", mgr.vector_transport_method)
    return print(io, ")")
end
function status_summary(mgr::MomentumGradientRule; context::Symbol = :default)
    (context === :short) && return repr(mgr)
    (context === :inline) && return "A momentum gradient direction processor with m=$(mgr.momentum) using $(mgr.vector_transport_method)"
    return """
    Momentum Gradient Rule

    ## Parameters
    * direction:              $(_MANOPT_INDENT)$(status_summary(mgr.direction; context = context))
    * momentum:               $(_MANOPT_INDENT)$(mgr.momentum)
    * vector transport method:$(_MANOPT_INDENT)$(mgr.vector_transport_method)"""
end

"""
    MomentumGradient(args...; kwargs...)

Append a momentum to a gradient processor.

The last direction and last iterate are stored and the new is composed as ``η_i = m*η_{i-1}' - s d_i``,
where ``sd_i`` is the current (inner) direction and ``η_{i-1}'`` is the vector transported
last direction multiplied by momentum ``m``.
This is the Riemannian version of gradient descent with momentum, first used in [RoyMhammediHarandi:2018; Section 3.1](@cite);
see [LeggioScuppa:2026; Section 6](@cite) for a convergence analysis.

# Input

* `M` (optional)

# Keyword arguments

$(_kwargs(:p))
* `direction=`[`IdentityUpdateRule`](@ref) preprocess the actual gradient before adding momentum
$(_kwargs(:X))
* `momentum=0.2` amount of momentum to use
$(_kwargs(:vector_transport_method))

$(_note(:ManifoldDefaultsFactory, "MomentumGradientRule"))
"""
function MomentumGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.MomentumGradientRule, args...; requires_point = true, kwargs...)
end

"""
    AverageGradientRule <: DirectionUpdateRule

Add an average of gradients to a gradient processor.

A set of previous directions (from the inner processor) and the last iterate are stored.
The average is taken after vector transporting them to the current iterates tangent space.


# Fields

* `gradients`:               the last `n` gradient/direction updates
* `last_iterate`:            last iterate (needed to transport the gradients)
* `direction`:               internal [`DirectionUpdateRule`](@ref) to determine directions to apply the averaging to
$(_kwargs(:vector_transport_method))

# Constructors

    AverageGradientRule(
        M::AbstractManifold;
        p::P=rand(M);
        n::Int=10
        direction::Union{<:DirectionUpdateRule,ManifoldDefaultsFactory}=IdentityUpdateRule(),
        gradients = fill(zero_vector(p.M, o.x),n),
        last_iterate = deepcopy(x0),
        vector_transport_method = default_vector_transport_method(M, typeof(p))
    )
    AverageGradientRule(M::AbstractManifold, p; kwargs...)

Add average to a gradient problem, where

* `n`:                       determines the size of averaging
* `direction`:               is the internal [`DirectionUpdateRule`](@ref) to determine the gradients to store
* `gradients`:               can be pre-filled with some history
* `last_iterate`:            stores the last iterate
$(_kwargs(:vector_transport_method))
"""
mutable struct AverageGradientRule{
        P, T, D <: DirectionUpdateRule, VTM <: AbstractVectorTransportMethod, A <: AbstractVector{<:T},
    } <: DirectionUpdateRule
    gradients::A
    last_iterate::P
    direction::D
    vector_transport_method::VTM
    function AverageGradientRule(;
            gradients::A, last_iterate::P, direction::D, vector_transport_method::VTM
        ) where {P, A <: AbstractVector, D <: DirectionUpdateRule, VTM <: AbstractVectorTransportMethod}
        return new{P, eltype(gradients), D, VTM, A}(gradients, last_iterate, direction, vector_transport_method)
    end
end
function AverageGradientRule(M::AbstractManifold, p; kwargs...)
    return AverageGradientRule(M; p = copy(M, p), kwargs...)
end
function AverageGradientRule(
        M::AbstractManifold;
        p::P = rand(M),
        n::Int = 10,
        direction::Union{<:DirectionUpdateRule, ManifoldDefaultsFactory} = Gradient(),
        gradients = [zero_vector(M, p) for _ in 1:n],
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
    ) where {P, VTM}
    dir = _produce_type(direction, M)
    return AverageGradientRule(;
        gradients = gradients, last_iterate = copy(M, p), direction = dir,
        vector_transport_method = vector_transport_method,
    )
end
function (a::AverageGradientRule)(
        mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
    )
    # remove oldest/last
    pop!(a.gradients)
    M = get_manifold(mp)
    p = get_iterate(s)
    _, d = a.direction(mp, s, k) #get inner gradient and step
    for g in a.gradients
        vector_transport_to!(M, g, a.last_iterate, g, p, a.vector_transport_method)
    end
    pushfirst!(a.gradients, copy(M, p, d))
    copyto!(M, a.last_iterate, p)
    return 1.0, 1 / length(a.gradients) .* sum(a.gradients)
end
function Base.show(io::IO, agr::AverageGradientRule)
    print(io, "AverageGradientRule(; gradients = ", agr.gradients)
    print(io, "last_iterate = ", agr.last_iterate, ", direction = ", agr.direction)
    print(io, "vector_transport_method = ", agr.vector_transport_method)
    return print(io, ")")
end
function status_summary(agr::AverageGradientRule; context::Symbol = :default)
    (context === :short) && return repr(agr)
    (context === :inline) && return "An average gradient direction processor with n=$(length(agr.gradients)) gradients to average over using $(agr.vector_transport_method)"
    return """
    Average Gradient Rule

    ## Parameters
    * direction:              $(_MANOPT_INDENT)$(status_summary(agr.direction; context = context))
    * number of gradients:    $(_MANOPT_INDENT)$(length(agr.gradients))
    * vector transport method:$(_MANOPT_INDENT)$(agr.vector_transport_method)"""
end
"""
    AverageGradient(; kwargs...)
    AverageGradient(M::AbstractManifold; kwargs...)

Add an average of gradients to a gradient processor. A set of previous directions (from the
inner processor) and the last iterate are stored, average is taken after vector transporting
them to the current iterates tangent space.

# Input

$(_args(:M)) (optional)

# Keyword arguments

$(_kwargs(:p; add_properties = [:as_Initial]))
* `direction=`[`IdentityUpdateRule`](@ref) preprocess the actual gradient before adding momentum
* `gradients=[zero_vector(M, p) for _ in 1:n]` how to initialize the internal storage
* `n=10` number of gradient evaluations to take the mean over
$(_kwargs([:X, :vector_transport_method]))

$(_note(:ManifoldDefaultsFactory, "AverageGradientRule"))
"""
function AverageGradient(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.AverageGradientRule, args...; requires_point = true, kwargs...)
end

@doc """
    NesterovRule <: DirectionUpdateRule

Compute a Nesterov inspired direction update rule.
See [`Nesterov`](@ref) for details

# Fields

* `γ::Real`, `μ::Real`: coefficients from the last iterate
* `v::P`:      an interim point to compute the next gradient evaluation point `y_k`
* `shrinkage`: a function `k -> ...` to compute the shrinkage ``β_k`` per iterate `k`.
$(_kwargs(:inverse_retraction_method))

# Constructor

    NesterovRule(M::AbstractManifold; kwargs...)
    NesterovRule(M::AbstractManifold, p; kwargs...)

## Keyword arguments

$(_kwargs(:p; add_properties = [:as_Initial]))
* `γ=0.001`
* `μ=0.9`
* `shrinkage = k -> 0.8`
$(_kwargs(:inverse_retraction_method))

# See also

[`Nesterov`](@ref)
"""
mutable struct NesterovRule{P, R <: Real, IRM <: AbstractInverseRetractionMethod, RM <: AbstractRetractionMethod, F} <: DirectionUpdateRule
    γ::R
    μ::R
    v::P
    shrinkage::F
    inverse_retraction_method::IRM
    retraction_method::RM
    function NesterovRule(;
            γ::R, μ::R, v::P, shrinkage::F, inverse_retraction_method::IRM, retraction_method::RM
        ) where {P, R <: Real, IRM <: AbstractInverseRetractionMethod, RM <: AbstractRetractionMethod, F}
        return new{P, R, IRM, RM, F}(γ, μ, v, shrinkage, inverse_retraction_method, retraction_method)
    end
end
function NesterovRule(M::AbstractManifold, p; kwargs...)
    return NesterovRule(M; p = copy(M, p), kwargs...)
end
function NesterovRule(
        M::AbstractManifold; p::P = rand(M), γ::Real = 0.001, μ::Real = 0.9, shrinkage::Function = i -> 0.8,
        inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(M, typeof(p)),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    ) where {P}
    # “Unify” the type of the two parameters, since they share a type parameter
    R = float(promote_type(typeof(γ), typeof(μ)))
    γ, μ = convert.(Ref(R), (γ, μ))
    p_ = maybe_wrap_variable(p)
    return NesterovRule(
        γ = γ, μ = μ, v = copy(M, p_), shrinkage = shrinkage, inverse_retraction_method = inverse_retraction_method, retraction_method = retraction_method,
    )
end
function (n::NesterovRule)(mp::AbstractManoptProblem, s::AbstractGradientSolverState, k)
    M = get_manifold(mp)
    h = get_stepsize(mp, s, k)
    p = get_iterate(s)
    α = (h * (n.μ - n.γ) + sqrt(h^2 * (n.μ - n.γ)^2 + 4 * h * n.γ)) / 2
    γbar = (1 - α) * n.γ + α * n.μ
    y = retract(
        M, p,
        ((α * n.γ) / (n.γ + α * n.μ)) * inverse_retract(M, p, n.v, n.inverse_retraction_method),
        n.retraction_method,
    )
    gradf_yk = get_gradient(mp, y)
    xn = retract(M, y, -h * gradf_yk, n.retraction_method)
    d =
        (((1 - α) * n.γ) / γbar) * inverse_retract(M, y, n.v, n.inverse_retraction_method) -
        (α / γbar) * gradf_yk
    retract!(M, n.v, y, d, n.retraction_method)
    n.γ = 1 / (1 + n.shrinkage(k)) * γbar
    return h, (-1 / h) * inverse_retract(M, p, xn, n.inverse_retraction_method) # outer update
end
function Base.show(io::IO, nr::NesterovRule)
    print(io, "NesterovRule(; γ = ", nr.γ, ", μ = ", nr.μ, ", v = ", nr.v, ", shrinkage = ", nr.shrinkage)
    return print(io, ", inverse_retraction_method = ", nr. inverse_retraction_method, ", retraction_method = ", nr.retraction_method, ")")
end
function status_summary(nr::NesterovRule; context::Symbol = :default)
    (context === :short) && return repr(nr)
    (context === :inline) && return "A Nesterov gradient direction processor using $(nr.retraction_method) and $(nr.inverse_retraction_method)"
    return """
    Nesterov Rule

    ## Parameters
    γ:                        $(_MANOPT_INDENT)$(nr.γ)
    μ:                        $(_MANOPT_INDENT)$(nr.μ)
    shrinkage:                $(_MANOPT_INDENT)$(nr.shrinkage)
    inverse_retraction_method:$(_MANOPT_INDENT)$(nr.inverse_retraction_method)
    retraction_method:        $(_MANOPT_INDENT)$(nr.retraction_method)
    """
end

@doc """
    Nesterov(; kwargs...)
    Nesterov(M::AbstractManifold; kwargs...)

Assume ``f`` is ``L``-Lipschitz and ``μ``-strongly convex. Given

* a step size ``h_k<$(_tex(:frac, "1", "L"))`` (from the [`GradientDescentState`](@ref))
* a `shrinkage` parameter ``β_k``
* and a current iterate ``p_k``
* as well as the interim values ``γ_k`` and ``v_k`` from the previous iterate.

This compute a Nesterov type update using the following steps, see [ZhangSra:2018](@cite)

1. Compute the positive root ``α_k∈(0,1)`` of ``α^2 = h_k$(_tex(:bigl))((1-α_k)γ_k+α_k μ$(_tex(:bigr)))``.
2. Set ``$(_tex(:bar, "γ"))_k+1 = (1-α_k)γ_k + α_kμ``
3. ``y_k = $(_tex(:retr))_{p_k}\\Bigl(\\frac{α_kγ_k}{γ_k + α_kμ}$(_tex(:retr))^{-1}_{p_k}v_k \\Bigr)``
4. ``x_{k+1} = $(_tex(:retr))_{y_k}(-h_k $(_tex(:grad))f(y_k))``
5. ``v_{k+1} = $(_tex(:retr))_{y_k}\\Bigl(\\frac{(1-α_k)γ_k}{$(_tex(:bar, "γ"))_k}$(_tex(:retr))_{y_k}^{-1}(v_k) - \\frac{α_k}{$(_tex(:bar, "γ"))_{k+1}}$(_tex(:grad))f(y_k) \\Bigr)``
6. ``γ_{k+1} = \\frac{1}{1+β_k}$(_tex(:bar, "γ"))_{k+1}``

Then the direction from ``p_k`` to ``p_k+1`` by ``d = $(_tex(:invretr))_{p_k}p_{k+1}`` is returned.

# Input

$(_args(:M)) (optional)

# Keyword arguments

$(_kwargs(:p; add_properties = [:as_Initial]))
* `γ=0.001`
* `μ=0.9`
* `shrinkage = k -> 0.8`
$(_kwargs(:inverse_retraction_method))

$(_note(:ManifoldDefaultsFactory, "NesterovRule"))
"""
function Nesterov(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.NesterovRule, args...; requires_point = true, kwargs...)
end

"""
    PreconditionedDirectionRule <: DirectionUpdateRule

Add a preconditioning as gradient processor, see [`PreconditionedDirection`](@ref)
for more mathematical background.

# Fields

* `direction`:      internal [`DirectionUpdateRule`](@ref) to determine directions to apply the preconditioning to
* `preconditioner`: the preconditioner function

# Constructors

    PreconditionedDirectionRule(
        M::AbstractManifold,
        preconditioner;
        direction::Union{<:DirectionUpdateRule,ManifoldDefaultsFactory}=IdentityUpdateRule(),
        evaluation::AbstractEvaluationType=AllocatingEvaluation()
    )

Add preconditioning to a gradient problem.

# Input

$(_args(:M))
* `preconditioner`:   preconditioner function, either as a `(M, Y, p, X) -> Y` mutating function

# Keyword arguments

* `direction=`[`IdentityUpdateRule`](@ref) internal [`DirectionUpdateRule`](@ref) to determine the gradients to store or a [`ManifoldDefaultsFactory`](@ref) generating one
"""
mutable struct PreconditionedDirectionRule{D <: DirectionUpdateRule, F} <: DirectionUpdateRule
    preconditioner::F
    direction::D
    function PreconditionedDirectionRule(;
            preconditioner::F, direction::D, evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        ) where {D <: DirectionUpdateRule, F}
        preconditioner_ = maybe_wrap_function(preconditioner, evaluation; result = :TangentVector)
        return new{D, typeof(preconditioner_)}(preconditioner_, direction)
    end
end
function PreconditionedDirectionRule(
        M::AbstractManifold,
        preconditioner::F;
        direction::Union{<:DirectionUpdateRule, ManifoldDefaultsFactory} = Gradient(),
        evaluation::E = AllocatingEvaluation(),
    ) where {E <: AbstractEvaluationType, F}
    dir = _produce_type(direction, M)
    return PreconditionedDirectionRule(; preconditioner = preconditioner, direction = dir, evaluation = evaluation)
end
function (pg::PreconditionedDirectionRule)(
        mp::AbstractManoptProblem, s::AbstractGradientSolverState, k
    )
    M = get_manifold(mp)
    p = get_iterate(s)
    step, dir = pg.direction(mp, s, k) # get inner direction and step size
    pg.preconditioner(M, dir, p, dir)
    return step, dir
end
function Base.show(io::IO, pg::PreconditionedDirectionRule)
    print(io, "PreconditionedDirectionRule(; direction = ", pg.direction, ", preconditioner = ", pg.preconditioner)
    return print(io, ")")
end
function status_summary(pg::PreconditionedDirectionRule; context::Symbol = :default)
    (context === :short) && return repr(pg)
    (context === :inline) && return "A preconditioner gradient processor"
    return """
    Preconditioned Direction Rule

    ## Parameters
    preconditioner: $(_MANOPT_INDENT)$(pg.preconditioner)

    ## Direction Rule
    $(_in_str(status_summary(pg.direction; context = context); indent = 1, headers = 1))
    """
end

"""
    PreconditionedDirection(preconditioner; kwargs...)
    PreconditionedDirection(M::AbstractManifold, preconditioner; kwargs...)

Add a preconditioner to a gradient processor following the [motivation for optimization](https://en.wikipedia.org/wiki/Preconditioner#Preconditioning_in_optimization),
as a linear invertible map ``P: $(_math(:TangentSpace)) → $(_math(:TangentSpace))`` that usually should be

* symmetric: ``⟨X, P(Y)⟩ = ⟨P(X), Y⟩``
* positive definite ``⟨X, P(X)⟩ > 0`` for ``X`` not the zero-vector

The gradient is then preconditioned as ``P(X)``, where ``X`` is either the
gradient of the objective or the result of a previous (internally stored) gradient processor.

For example if you provide as the preconditioner the inverse of the Hessian ``$(_tex(:Hess))^{-1} f``,
you turn a gradient descent into a Newton method.

# Arguments

$(_args(:M)) (optional)
* `preconditioner`:   preconditioner function, either as a `(M, p, X) -> Y` allocating or `(M, Y, p, X) -> Y` mutating function

# Keyword arguments

* `direction=`[`IdentityUpdateRule`](@ref) internal [`DirectionUpdateRule`](@ref) to determine the gradients to store or a [`ManifoldDefaultsFactory`](@ref) generating one
$(_kwargs(:evaluation))

$(_note(:ManifoldDefaultsFactory, "PreconditionedDirectionRule"))
"""
function PreconditionedDirection(args...; kwargs...)
    return ManifoldDefaultsFactory(Manopt.PreconditionedDirectionRule, args...; kwargs...)
end
