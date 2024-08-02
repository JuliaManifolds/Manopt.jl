
"""
    AbstractConstrainedFunctor{T}

A common supertype for fucntors that model constraint functions.

This supertype provides access for the fields ``λ`` and ``μ``, the dual variables of
constraintsnof type `T`.
"""
abstract type AbstractConstrainedFunctor{T} end

function set_manopt_parameter!(
    acf::AbstractConstrainedFunctor{T}, ::Val{:μ}, μ::T
) where {T}
    acf.μ = μ
    return acf
end
get_manopt_parameter(acf::AbstractConstrainedFunctor, ::Val{:μ}) = acf.μ
function set_manopt_parameter!(
    acf::AbstractConstrainedFunctor{T}, ::Val{:λ}, λ::T
) where {T}
    acf.λ = λ
    return acf
end
get_manopt_parameter(acf::AbstractConstrainedFunctor, ::Val{:λ}) = acf.λ

"""
    AbstractConstrainedSlackFunctor{T,R}

A common supertype for fucntors that model constraint functions with slack.

This supertype additionally provides access for the fields
* `μ::T` the dual for the inequality constraints
* `s::T` the slack parametyer, and
* `β::R` the  the barrier parameter
which is also of typee `T`.
"""
abstract type AbstractConstrainedSlackFunctor{T,R} end

function set_manopt_parameter!(
    acsf::AbstractConstrainedSlackFunctor{T}, ::Val{:s}, s::T
) where {T}
    acsf.s = s
    return acsf
end
get_manopt_parameter(acsf::AbstractConstrainedSlackFunctor, ::Val{:s}) = acsf.s
function set_manopt_parameter!(
    acsf::AbstractConstrainedSlackFunctor{T}, ::Val{:μ}, μ::T
) where {T}
    acsf.μ = μ
    return acsf
end
get_manopt_parameter(acsf::AbstractConstrainedSlackFunctor, ::Val{:μ}) = acsf.μ
function set_manopt_parameter!(
    acsf::AbstractConstrainedSlackFunctor{T,R}, ::Val{:β}, β::R
) where {T,R}
    acsf.β = β
    return acsf
end
get_manopt_parameter(acsf::AbstractConstrainedSlackFunctor, ::Val{:β}) = acsf.β

@doc raw"""
    ConstrainedManifoldObjective{T<:AbstractEvaluationType, C<:ConstraintType} <: AbstractManifoldObjective{T}

Describes the constrained objective
```math
\begin{aligned}
 \operatorname*{arg\,min}_{p ∈\mathcal{M}} & f(p)\\
 \text{subject to } &g_i(p)\leq0 \quad \text{ for all } i=1,…,m,\\
 \quad &h_j(p)=0 \quad \text{ for all } j=1,…,n.
\end{aligned}
```

# Fields

* `objective`: an [`AbstractManifoldObjective`](@ref) representing the unconstrained
  objective, that is containing cost ``f``, the gradient of the cost ``f`` and maybe the Hessian.
* `equality_constraints`: an [`AbstractManifoldObjective`](@ref) representing the equality constraints
``h: \mathcal M → \mathbb R^n`` also possibly containing its gradient and/or Hessian
* `equality_constraints`: an [`AbstractManifoldObjective`](@ref) representing the equality constraints
``h: \mathcal M → \mathbb R^n`` also possibly containing its gradient and/or Hessian

# Constructors
    ConstrainedManifoldObjective(M::AbstractManifold, f, grad_f;
        g=nothing,
        grad_g=nothing,
        h=nothing,
        grad_h=nothing;
        hess_f=nothing,
        hess_g=nothing,
        hess_h=nothing,
        equality_constraints=nothing,
        inequality_constraints=nothing,
        evaluation=AllocatingEvaluation(),
        M = nothing,
        p = isnothing(M) ? nothing : rand(M),
    )

Generate the constrained objective based on all involved single functions `f`, `grad_f`, `g`,
`grad_g`, `h`, `grad_h`, and optionally a Hessian for each of these.
With `equality_constraints` and `inequality_constraints` you have to provide the dimension
of the ranges of `h` and `g`, respectively.
You can also provide a manifold `M` and a point `p` to use one evaluation of the constraints
to automatically try to determine these sizes.

    ConstrainedManifoldObjective(M::AbstractManifold, mho::AbstractManifoldObjective;
        equality_constraints = nothing,
        inequality_constraints = nothing
    )

Generate the constrained objective either with explicit constraints ``g`` and ``h``, and
their gradients, or in the form where these are already encapsulated in [`VectorGradientFunction`](@ref)s.

Both variants require that at least one of the constraints (and its gradient) is provided.
If any of the three parts provides a Hessian, the corresponding object, that is a
[`ManifoldHessianObjective`](@ref) for `f` or a [`VectorHessianFunction`](@ref) for `g` or `h`,
respectively, is created.
"""
struct ConstrainedManifoldObjective{
    T<:AbstractEvaluationType,
    MO<:AbstractManifoldObjective,
    EMO<:Union{AbstractVectorGradientFunction,Nothing},
    IMO<:Union{AbstractVectorGradientFunction,Nothing},
} <: AbstractManifoldObjective{T}
    objective::MO
    equality_constraints::EMO
    inequality_constraints::IMO
end
function _vector_function_type_hint(f)
    (!isnothing(f) && isa(f, AbstractVector)) && return ComponentVectorialType()
    return FunctionVectorialType()
end

function _val_to_ncons(val)
    sv = size(val)
    if sv === ()
        return 1
    else
        return sv[end]
    end
end

# Try to infer the number of constraints
function _number_of_constraints(
    g,
    grad_g;
    function_type::Union{AbstractVectorialType,Nothing}=nothing,
    jacobian_type::Union{AbstractVectorialType,Nothing}=nothing,
    M::Union{AbstractManifold,Nothing}=nothing,
    p=isnothing(M) ? nothing : rand(M),
)
    if !isnothing(g)
        if isa(function_type, ComponentVectorialType) || isa(g, AbstractVector)
            return length(g)
        end
    end
    if !isnothing(grad_g)
        if isa(jacobian_type, ComponentVectorialType) || isa(grad_g, AbstractVector)
            return length(grad_g)
        end
    end
    # These are more expensive, since they evaluate and hence allocate
    if !isnothing(M) && !isnothing(p)
        # For functions on vector representations, the last size is equal to length
        # on array power manifolds, this also yields the number of elements
        (!isnothing(g)) && (return _val_to_ncons(g(M, p)))
        (!isnothing(grad_g)) && (return _val_to_ncons(grad_g(M, p)))
    end
    return -1
end

function ConstrainedManifoldObjective(
    f,
    grad_f,
    g,
    grad_g,
    h,
    grad_h;
    hess_f=nothing,
    hess_g=nothing,
    hess_h=nothing,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    equality_type::AbstractVectorialType=_vector_function_type_hint(h),
    equality_gradient_type::AbstractVectorialType=_vector_function_type_hint(grad_h),
    equality_hessian_type::AbstractVectorialType=_vector_function_type_hint(hess_h),
    inequality_type::AbstractVectorialType=_vector_function_type_hint(g),
    inequality_gradient_type::AbstractVectorialType=_vector_function_type_hint(grad_g),
    inequality_hessian_type::AbstractVectorialType=_vector_function_type_hint(hess_g),
    equality_constraints::Union{Integer,Nothing}=nothing,
    inequality_constraints::Union{Integer,Nothing}=nothing,
    M::Union{AbstractManifold,Nothing}=nothing,
    p=isnothing(M) ? nothing : rand(M),
    kwargs...,
)
    if isnothing(hess_f)
        objective = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    else
        objective = ManifoldHessianObjective(f, grad_f, hess_f; evaluation=evaluation)
    end
    num_eq = isnothing(equality_constraints) ? -1 : equality_constraints
    if isnothing(h) || isnothing(grad_h)
        eq = nothing
    else
        if isnothing(equality_constraints)
            # try to guess
            num_eq = _number_of_constraints(
                h,
                grad_h;
                function_type=equality_type,
                jacobian_type=equality_gradient_type,
                M=M,
                p=p,
            )
        end
        # if it is still < 0, this can not be used
        (num_eq < 0) && error(
            "Please specify a positive number of `equality_constraints` (provided $(equality_constraints))",
        )
        if isnothing(hess_h)
            eq = VectorGradientFunction(
                h,
                grad_h,
                num_eq;
                evaluation=evaluation,
                function_type=equality_type,
                jacobian_type=equality_gradient_type,
            )
        else
            eq = VectorHessianFunction(
                h,
                grad_h,
                hess_h,
                num_eq;
                evaluation=evaluation,
                function_type=equality_type,
                jacobian_type=equality_gradient_type,
                hessian_type=equality_hessian_type,
            )
        end
    end
    num_ineq = isnothing(inequality_constraints) ? -1 : inequality_constraints
    if isnothing(g) || isnothing(grad_g)
        ineq = nothing
    else
        if isnothing(inequality_constraints)
            # try to guess
            num_ineq = _number_of_constraints(
                g,
                grad_g;
                function_type=inequality_type,
                jacobian_type=inequality_gradient_type,
                M=M,
                p=p,
            )
        end
        # if it is still < 0, this can not be used
        (num_ineq < 0) && error(
            "Please specify a positive number of `inequality_constraints` (provided $(inequality_constraints))",
        )
        if isnothing(hess_g)
            ineq = VectorGradientFunction(
                g,
                grad_g,
                num_ineq;
                evaluation=evaluation,
                function_type=inequality_type,
                jacobian_type=inequality_gradient_type,
            )
        else
            ineq = VectorHessianFunction(
                g,
                grad_g,
                hess_g,
                num_ineq;
                evaluation=evaluation,
                function_type=inequality_type,
                jacobian_type=inequality_gradient_type,
                hessian_type=inequality_hessian_type,
            )
        end
    end
    return ConstrainedManifoldObjective(
        objective; equality_constraints=eq, inequality_constraints=ineq
    )
end
function ConstrainedManifoldObjective(
    objective::MO;
    equality_constraints::EMO=nothing,
    inequality_constraints::IMO=nothing,
    kwargs...,
) where {E<:AbstractEvaluationType,MO<:AbstractManifoldObjective{E},IMO,EMO}
    if isnothing(equality_constraints) && isnothing(inequality_constraints)
        throw(ErrorException("""
        Neither the inequality and the equality constraints are provided.
        You can not generate a `ConstrainedManifoldObjective` without actual
        constraints.

        If you do not have any constraints, you could also take the `objective`
        (probably `f` and `grad_f`) and work with an unconstrained solver.
        """))
    end
    return ConstrainedManifoldObjective{E,MO,EMO,IMO}(
        objective, equality_constraints, inequality_constraints
    )
end
function ConstrainedManifoldObjective(
    f, grad_f; g=nothing, grad_g=nothing, h=nothing, grad_h=nothing, kwargs...
)
    return ConstrainedManifoldObjective(f, grad_f, g, grad_g, h, grad_h; kwargs...)
end

@doc raw"""
    ConstrainedProblem{
        TM <: AbstractManifold,
        O <: AbstractManifoldObjective
        HR<:Union{AbstractPowerRepresentation,Nothing},
        GR<:Union{AbstractPowerRepresentation,Nothing},
        HHR<:Union{AbstractPowerRepresentation,Nothing},
        GHR<:Union{AbstractPowerRepresentation,Nothing},
    } <: AbstractManoptProblem{TM}

A constrained problem might feature different ranges for the
(vectors of) gradients of the equality and inequality constraints.

The ranges are required in a few places to allocate memory and access elements
correctly, they work as follows:

Assume the objective is
```math
\begin{aligned}
 \operatorname*{arg\,min}_{p ∈\mathcal{M}} & f(p)\\
 \text{subject to } &g_i(p)\leq0 \quad \text{ for all } i=1,…,m,\\
 \quad &h_j(p)=0 \quad \text{ for all } j=1,…,n.
\end{aligned}
```

then the gradients can (classically) be considered as vectors of the
components gradients, for example
``\bigl(\operatorname{grad} g_1(p), \operatorname{grad} g_2(p), …, \operatorname{grad} g_m(p) \bigr)``.

In another interpretation, this can be considered a point on the tangent space
at ``P = (p,…,p) \in \mathcal M^m``, so in the tangent space to the [`PowerManifold`](@extref `ManifoldsBase.PowerManifold`) ``\mathcal M^m``.
The case where this is a [`NestedPowerRepresentation`](@extref) this agrees with the
interpretation from before, but on power manifolds, more efficient representations exist.

To then access the elements, the range has to be specified. That is what this
problem is for.

# Constructor
    ConstrainedManoptProblem(
        M::AbstractManifold,
        co::ConstrainedManifoldObjective;
        range=NestedPowerRepresentation(),
        gradient_equality_range=range,
        gradient_inequality_range=range
        hessian_equality_range=range,
        hessian_inequality_range=range
    )

Creates a constrained Manopt problem specifying an [`AbstractPowerRepresentation`](@extref ManifoldsBase.AbstractPowerRepresentation)
for both the `gradient_equality_range` and the `gradient_inequality_range`, respectively.
"""
struct ConstrainedManoptProblem{
    TM<:AbstractManifold,
    O<:AbstractManifoldObjective,
    HR<:Union{AbstractPowerRepresentation,Nothing},
    GR<:Union{AbstractPowerRepresentation,Nothing},
    HHR<:Union{AbstractPowerRepresentation,Nothing},
    GHR<:Union{AbstractPowerRepresentation,Nothing},
} <: AbstractManoptProblem{TM}
    manifold::TM
    grad_equality_range::HR
    grad_inequality_range::GR
    hess_equality_range::HHR
    hess_inequality_range::GHR
    objective::O
end

function ConstrainedManoptProblem(
    M::TM,
    objective::O;
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
    gradient_equality_range::HR=range,
    gradient_inequality_range::GR=range,
    hessian_equality_range::HHR=range,
    hessian_inequality_range::GHR=range,
) where {
    TM<:AbstractManifold,
    O<:AbstractManifoldObjective,
    GR<:Union{AbstractPowerRepresentation,Nothing},
    HR<:Union{AbstractPowerRepresentation,Nothing},
    GHR<:Union{AbstractPowerRepresentation,Nothing},
    HHR<:Union{AbstractPowerRepresentation,Nothing},
}
    return ConstrainedManoptProblem{TM,O,HR,GR,HHR,GHR}(
        M,
        gradient_equality_range,
        gradient_inequality_range,
        hessian_equality_range,
        hessian_inequality_range,
        objective,
    )
end
get_manifold(cmp::ConstrainedManoptProblem) = cmp.manifold
get_objective(cmp::ConstrainedManoptProblem) = cmp.objective

@doc raw"""
    LagrangianCost{CO,T} <: AbstractConstrainedFunctor{T}

Implement the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`.

```math
\mathcal L(p; μ, λ)
= f(p) +  \sum_{i=1}^m μ_ig_i(p) + \sum_{j=1}^n λ_jh_j(p)
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianCost(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the Lagrangian ``\mathcal L``
you can also call

```
LagrangianCost(co, μ, λ)(M,p)
```
"""
mutable struct LagrangianCost{CO,T} <: AbstractConstrainedFunctor{T}
    co::CO
    μ::T
    λ::T
end
function (lc::LagrangianCost)(M, p)
    c = get_cost(M, lc.co, p)
    g = get_inequality_constraint(M, lc.co, p, :)
    h = get_equality_constraint(M, lc.co, p, :)
    (length(g) > 0) && (c += sum(lc.μ .* g))
    (length(h) > 0) && (c += sum(lc.λ .* h))
    return c
end
function show(io::IO, lc::LagrangianCost)
    return print(io, "LagrangianCost\n\twith μ=$(lc.μ), λ=$(lc.λ)")
end

@doc raw"""
    LagrangianGradient{CO,T}

The gradient of the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`
with respect to the variable ``p``. The formula reads

```math
\operatorname{grad}_p \mathcal L(p; μ, λ)
= \operatorname{grad} f(p) +  \sum_{i=1}^m μ_i \operatorname{grad} g_i(p) + \sum_{j=1}^n λ_j \operatorname{grad} h_j(p)
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianGradient(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the gradient of the Lagrangian ``\operatorname{grad}_p \mathcal L``
you can also call `LagrangianGradient(co, μ, λ)(M,p)` or `LagrangianGradient(co, μ, λ)(M,X,p)` for the in-place variant.
"""
mutable struct LagrangianGradient{CO,T} <: AbstractConstrainedFunctor{T}
    co::CO
    μ::T
    λ::T
end
function (lg::LagrangianGradient)(M, p)
    X = zero_vector(M, p)
    return lg(M, X, p)
end
function (lg::LagrangianGradient)(M, X, p)
    Y = copy(M, p, X)
    get_gradient!(M, X, lg.co, p)
    m = inequality_constraints_length(lg.co)
    n = equality_constraints_length(lg.co)
    for i in 1:m
        get_grad_inequality_constraint!(M, Y, lg.co, p, i)
        copyto!(M, X, p, X + lg.μ[i] * Y)
    end
    for j in 1:n
        get_grad_equality_constraint!(M, Y, lg.co, p, j)
        copyto!(M, X, p, X + lg.λ[j] * Y)
    end
    return X
end
function show(io::IO, lg::LagrangianGradient)
    return print(io, "LagrangianGradient\n\twith μ=$(lg.μ), λ=$(lg.λ)")
end

@doc raw"""
    LagrangianHessian{CO, V, T}

The Hesian of the Lagrangian of a [`ConstrainedManifoldObjective`](@ref) `co`
with respect to the variable ``p``. The formula reads

```math
\operatorname{Hess}_p \mathcal L(p; μ, λ)[X]
= \operatorname{Hess} f(p) +  \sum_{i=1}^m μ_i \operatorname{Hess} g_i(p)[X] + \sum_{j=1}^n λ_j \operatorname{Hess} h_j(p)[X]
```

# Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned, where `T` represents a vector type.

# Constructor

    LagrangianHessian(co, μ, λ)

Create a functor for the Lagrangian with fixed dual variables.

# Example

When you directly want to evaluate the Hessian of the Lagrangian ``\operatorname{Hess}_p \mathcal L``
you can also call `LagrangianHessian(co, μ, λ)(M, p, X)` or `LagrangianHessian(co, μ, λ)(M, Y, p, X)` for the in-place variant.
"""
mutable struct LagrangianHessian{CO,T} <: AbstractConstrainedFunctor{T}
    co::CO
    μ::T
    λ::T
end
function (lH::LagrangianHessian)(M, p, X)
    Y = zero_vector(M, p)
    return lH(M, Y, p, X)
end
function (lH::LagrangianHessian)(M, Y, p, X)
    Z = copy(M, p, X)
    get_hessian!(M, Y, lH.co, p, X)
    n = inequality_constraints_length(lH.co)
    m = equality_constraints_length(lH.co)
    for i in 1:n
        get_hess_inequality_constraint!(M, Z, lH.co, p, X, i)
        copyto!(M, Y, p, Y + lH.μ[i] * Z)
    end
    for j in 1:m
        get_hess_equality_constraint!(M, Z, lH.co, p, X, j)
        copyto!(M, Y, p, Y + lH.λ[j] * Z)
    end
    return Y
end
function show(io::IO, lh::LagrangianHessian)
    return print(io, "LagrangianHessian\n\twith μ=$(lh.μ), λ=$(lh.λ)")
end

@doc raw"""
    equality_constraints_length(co::ConstrainedManifoldObjective)

Return the number of equality constraints of an [`ConstrainedManifoldObjective`](@ref).
This acts transparently through [`AbstractDecoratedManifoldObjective`](@ref)s
"""
function equality_constraints_length(co::ConstrainedManifoldObjective)
    return isnothing(co.equality_constraints) ? 0 : length(co.equality_constraints)
end
function equality_constraints_length(co::AbstractDecoratedManifoldObjective)
    return equality_constraints_length(get_objective(co, false))
end

@doc raw"""
    get_unconstrained_objective(co::ConstrainedManifoldObjective)

Returns the internally stored unconstrained [`AbstractManifoldObjective`](@ref)
within the [`ConstrainedManifoldObjective`](@ref).
"""
get_unconstrained_objective(co::ConstrainedManifoldObjective) = co.objective

function get_constraints(mp::AbstractManoptProblem, p)
    Base.depwarn(
        "get_constraints will be removed in a future release, use `get_equality_constraint($mp, $p, :)` and `get_inequality_constraint($mp, $p, :)`, respectively",
        :get_constraints,
    )
    return [
        get_inequality_constraint(get_manifold(mp), get_objective(mp), p, :),
        get_equality_constraint(get_manifold(mp), get_objective(mp), p, :),
    ]
end
function get_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    Base.depwarn(
        "get_constraints will be removed in a future release, use `get_equality_constraint($M, $co, $p, :)` and `get_inequality_constraint($M, $co, $p, :)`, respectively",
        :get_constraints,
    )
    return [get_inequality_constraint(M, co, p, :), get_equality_constraint(M, co, p, :)]
end

function get_cost(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    return get_cost(M, co.objective, p)
end
function get_cost_function(co::ConstrainedManifoldObjective, recursive=false)
    return get_cost_function(co.objective, recursive)
end

Base.@deprecate get_equality_constraints(amp::AbstractManoptProblem, p) get_equality_constraint(
    amp, p, :,
)

Base.@deprecate get_equality_constraints!(amp::AbstractManoptProblem, X, p) get_equality_constraint!(
    amp, X, p, :,
)

Base.@deprecate get_equality_constraints(
    M::AbstractManifold, co::AbstractManifoldObjective, p
) get_equality_constraint(M, co, p, :)

Base.@deprecate get_equality_constraints!(
    M::AbstractManifold, X, co::AbstractManifoldObjective, p
) get_equality_constraint!(M, X, co, p, :)

@doc raw"""
    get_equality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_equality_constraint(M::AbstractManifold, objective, p, j=:)

Evaluate equality constraints of a [`ConstrainedManifoldObjective`](@ref) `objective`
at point `p` and indices `j` (by default `:` which corresponds to all indices).
"""
function get_equality_constraint end

function get_equality_constraint(mp::AbstractManoptProblem, p, j=:)
    return get_equality_constraint(get_manifold(mp), get_objective(mp), p, j)
end

function get_equality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, j=:
)
    return get_equality_constraint(M, get_objective(admo, false), p, j)
end

function get_equality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:
)
    if isnothing(co.equality_constraints)
        return number_eltype(p)[]
    else
        return get_value(M, co.equality_constraints, p, j)
    end
end

function get_gradient(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    return get_gradient(M, co.objective, p)
end
function get_gradient!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p)
    return get_gradient!(M, X, co.objective, p)
end
function get_gradient_function(co::ConstrainedManifoldObjective, recursive=false)
    return get_gradient_function(co.objective, recursive)
end

Base.@deprecate get_inequality_constraints(amp::AbstractManoptProblem, p) get_inequality_constraint(
    amp, p, :,
)
Base.@deprecate get_inequality_constraints(
    M::AbstractManifold, co::AbstractManifoldObjective, p
) get_inequality_constraint(M, co, p, :)

@doc raw"""
    get_inequality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())

Evaluate inequality constraints of a [`ConstrainedManifoldObjective`](@ref) `objective`
at point `p` and indices `j` (by default `:` which corresponds to all indices).
"""
function get_inequality_constraint end

function get_inequality_constraint(mp::AbstractManoptProblem, p, j=:)
    return get_inequality_constraint(get_manifold(mp), get_objective(mp), p, j)
end
function get_inequality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, j=:
)
    return get_inequality_constraint(M, get_objective(admo, false), p, j)
end
function get_inequality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:
)
    if isnothing(co.inequality_constraints)
        return number_eltype(p)[]
    else
        return get_value(M, co.inequality_constraints, p, j)
    end
end

@doc raw"""
    get_grad_equality_constraint(amp::AbstractManoptProblem, p, j)
    get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())
    get_grad_equality_constraint!(amp::AbstractManoptProblem, X, p, j)
    get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())

Evaluate the gradient or gradients  of the equality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""
function get_grad_equality_constraint end

function get_grad_equality_constraint(
    amp::AbstractManoptProblem,
    p,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    return get_grad_equality_constraint(get_manifold(amp), get_objective(amp), p, j, range)
end
function get_grad_equality_constraint(cmp::ConstrainedManoptProblem, p, j=:)
    return get_grad_equality_constraint(
        get_manifold(cmp), get_objective(cmp), p, j, cmp.grad_equality_range
    )
end
function get_grad_equality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_grad_equality_constraint(M, get_objective(admo, false), args...)
end
function get_grad_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    if isnothing(co.equality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_gradient(M, co.equality_constraints, p, j, range)
end

function get_grad_equality_constraint!(
    amp::AbstractManoptProblem,
    X,
    p,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    return get_grad_equality_constraint!(
        get_manifold(amp), X, get_objective(amp), p, j, range
    )
end
function get_grad_equality_constraint!(cmp::ConstrainedManoptProblem, X, p, j=:)
    return get_grad_equality_constraint!(
        get_manifold(cmp), X, get_objective(cmp), p, j, cmp.grad_equality_range
    )
end
function get_grad_equality_constraint!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_grad_equality_constraint!(M, X, get_objective(admo, false), args...)
end

function get_grad_equality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective,
    p,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    isnothing(co.equality_constraints) && (return X)
    return get_gradient!(M, X, co.equality_constraints, p, j, range)
end

# Deprecate plurals
Base.@deprecate get_grad_equality_constraints(mp::AbstractManoptProblem, p) get_grad_equality_constraint(
    mp, p, :,
)
Base.@deprecate get_grad_equality_constraints(
    M::AbstractManifold, co::AbstractManifoldObjective, p
) get_grad_equality_constraint(M, co, p, :)
Base.@deprecate get_grad_equality_constraints!(mp::AbstractManoptProblem, X, p) get_grad_equality_constraint!(
    mp, X, p, :,
)
Base.@deprecate get_grad_equality_constraints!(
    M::AbstractManifold, X, co::AbstractManifoldObjective, p
) get_grad_equality_constraint!(M, X, co, p, :)

@doc raw"""
    get_grad_inequality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())
    get_grad_inequality_constraint!(amp::AbstractManoptProblem, X, p, j=:)
    get_grad_inequality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())

Evaluate the gradient or gradients of the inequality constraint ``(\operatorname{grad} g(p))_j`` or ``\operatorname{grad} g_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""
function get_grad_inequality_constraint end

function get_grad_inequality_constraint(
    amp::AbstractManoptProblem,
    p,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    return get_grad_inequality_constraint(
        get_manifold(amp), get_objective(amp), p, j, range
    )
end
function get_grad_inequality_constraint(cmp::ConstrainedManoptProblem, p, j=:)
    return get_grad_inequality_constraint(
        get_manifold(cmp), get_objective(cmp), p, j, cmp.grad_inequality_range
    )
end
function get_grad_inequality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_grad_inequality_constraint(M, get_objective(admo, false), args...)
end

function get_grad_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    if isnothing(co.inequality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_gradient(M, co.inequality_constraints, p, j, range)
end

function get_grad_inequality_constraint!(amp::AbstractManoptProblem, X, p, j)
    return get_grad_inequality_constraint!(get_manifold(amp), X, get_objective(amp), p, j)
end
function get_grad_inequality_constraint!(cmp::ConstrainedManoptProblem, X, p, j)
    return get_grad_inequality_constraint!(
        get_manifold(cmp), X, get_objective(cmp), p, j, cmp.grad_inequality_range
    )
end
function get_grad_inequality_constraint!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_grad_inequality_constraint!(M, X, get_objective(admo, false), args...)
end
function get_grad_inequality_constraint!(
    M::AbstractManifold,
    X,
    co::ConstrainedManifoldObjective,
    p,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    isnothing(co.inequality_constraints) && (return X)
    return get_gradient!(M, X, co.inequality_constraints, p, j, range)
end

#Deprecate plurals
Base.@deprecate get_grad_inequality_constraints(mp::AbstractManoptProblem, p) get_grad_inequality_constraint(
    mp, p, :,
)
Base.@deprecate get_grad_inequality_constraints(
    M::AbstractManifold, co::AbstractManifoldObjective, p
) get_grad_inequality_constraint(M, co, p, :)
Base.@deprecate get_grad_inequality_constraints!(mp::AbstractManoptProblem, X, p) get_grad_inequality_constraint!(
    mp, X, p, :,
)
Base.@deprecate get_grad_inequality_constraints!(
    M::AbstractManifold, X, co::AbstractManifoldObjective, p
) get_grad_inequality_constraint!(M, X, co, p, :)

function get_hessian(M::AbstractManifold, co::ConstrainedManifoldObjective, p, X)
    return get_hessian(M, co.objective, p, X)
end
function get_hessian!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X)
    return get_hessian!(M, Y, co.objective, p, X)
end
function get_hessian_function(co::ConstrainedManifoldObjective, recursive=false)
    return get_hessian_function(co.objective, recursive)
end

@doc raw"""
    get_hess_equality_constraint(amp::AbstractManoptProblem, p, j=:)
    get_hess_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())
    get_hess_equality_constraint!(amp::AbstractManoptProblem, X, p, j=:)
    get_hess_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j, range=NestedPowerRepresentation())

Evaluate the Hessian or Hessians of the equality constraint ``(\operatorname{Hess} h(p))_j`` or ``\operatorname{Hess} h_j(p)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the Hessian.
"""
function get_hess_equality_constraint end

function get_hess_equality_constraint(amp::AbstractManoptProblem, p, X, j=:)
    return get_hess_equality_constraint(get_manifold(amp), get_objective(amp), p, X, j)
end
function get_hess_equality_constraint(cmp::ConstrainedManoptProblem, p, X, j=:)
    return get_hess_equality_constraint(
        get_manifold(cmp), get_objective(cmp), p, X, j, cmp.hess_equality_range
    )
end
function get_hess_equality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_hess_equality_constraint(M, get_objective(admo, false), args...)
end
function get_hess_equality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p,
    X,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    if isnothing(co.equality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_hessian(M, co.equality_constraints, p, X, j, range)
end

function get_hess_equality_constraint!(
    amp::AbstractManoptProblem,
    Y,
    p,
    X,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    return get_hess_equality_constraint!(
        get_manifold(amp), Y, get_objective(amp), p, X, j, range
    )
end
function get_hess_equality_constraint!(cmp::ConstrainedManoptProblem, Y, p, X, j=:)
    return get_hess_equality_constraint!(
        get_manifold(cmp), Y, get_objective(cmp), p, X, j, cmp.hess_equality_range
    )
end
function get_hess_equality_constraint!(
    M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_hess_equality_constraint!(M, Y, get_objective(admo, false), args...)
end

function get_hess_equality_constraint!(
    M::AbstractManifold,
    Y,
    co::ConstrainedManifoldObjective,
    p,
    X,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    isnothing(co.equality_constraints) && (return Y)
    return get_hessian!(M, Y, co.equality_constraints, p, X, j, range)
end

@doc raw"""
    get_hess_inequality_constraint(amp::AbstractManoptProblem, p, X, j=:)
    get_hess_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, j=:, range=NestedPowerRepresentation())
    get_hess_inequality_constraint!(amp::AbstractManoptProblem, Y, p, j=:)
    get_hess_inequality_constraint!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, j=:, range=NestedPowerRepresentation())

Evaluate the Hessian or Hessians of the inequality constraint ``(\operatorname{Hess} g(p)[X])_j`` or ``\operatorname{Hess} g_j(p)[X]``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the Hessian.
"""
function get_hess_inequality_constraint end

function get_hess_inequality_constraint(
    amp::AbstractManoptProblem,
    p,
    X,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    return get_hess_inequality_constraint(
        get_manifold(amp), get_objective(amp), p, X, j, range
    )
end
function get_hess_inequality_constraint(cmp::ConstrainedManoptProblem, p, X, j=:)
    return get_hess_inequality_constraint(
        get_manifold(cmp), get_objective(cmp), p, X, j, cmp.hess_inequality_range
    )
end
function get_hess_inequality_constraint(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_hess_inequality_constraint(M, get_objective(admo, false), args...)
end

function get_hess_inequality_constraint(
    M::AbstractManifold,
    co::ConstrainedManifoldObjective,
    p,
    X,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    if isnothing(co.inequality_constraints)
        pM = PowerManifold(M, range, 0)
        q = rand(pM) # an empty vector or matrix
        return zero_vector(pM, q) # an empty vector or matrix of correct type
    end
    return get_hessian(M, co.inequality_constraints, p, X, j, range)
end

function get_hess_inequality_constraint!(
    amp::AbstractManoptProblem,
    Y,
    p,
    X,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    return get_hess_inequality_constraint!(
        get_manifold(amp), Y, get_objective(amp), p, X, j, range
    )
end
function get_hess_inequality_constraint!(cmp::ConstrainedManoptProblem, Y, p, X, j=:)
    return get_hess_inequality_constraint!(
        get_manifold(cmp), Y, get_objective(cmp), p, X, j, cmp.hess_inequality_range
    )
end
function get_hess_inequality_constraint!(
    M::AbstractManifold, Y, admo::AbstractDecoratedManifoldObjective, args...
)
    return get_hess_inequality_constraint!(M, Y, get_objective(admo, false), args...)
end
function get_hess_inequality_constraint!(
    M::AbstractManifold,
    Y,
    co::ConstrainedManifoldObjective,
    p,
    X,
    j=:,
    range::AbstractPowerRepresentation=NestedPowerRepresentation(),
)
    isnothing(co.inequality_constraints) && (return X)
    return get_hessian!(M, Y, co.inequality_constraints, p, X, j, range)
end

@doc raw"""
    inequality_constraints_length(cmo::ConstrainedManifoldObjective)

Return the number of inequality constraints of an [`ConstrainedManifoldObjective`](@ref) `cmo`.
This acts transparently through [`AbstractDecoratedManifoldObjective`](@ref)s
"""
function inequality_constraints_length(cmo::ConstrainedManifoldObjective)
    return isnothing(cmo.inequality_constraints) ? 0 : length(cmo.inequality_constraints)
end
function inequality_constraints_length(admo::AbstractDecoratedManifoldObjective)
    return inequality_constraints_length(get_objective(admo, false))
end

@doc raw"""
    is_feasible(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, kwargs...)

Evaluate whether a boint `p` on `M` is feasible with respect to the [`ConstrainedManifoldObjective`](@ref) `cmo`.
That is for the provided inequality constaints ``g: \mathcal M → ℝ^m`` and equality constaints ``h: \mathcal M \to ℝ^m``
from within `cmo`, the point ``p ∈ \mathcal M`` is feasible if
```math
g_i(p) ≤ 0, \text{ for all } i=1,…,m\quad\text{ and }\quad h_j(p) = 0, \text{ for all } j=1,…,n.
```

# Keyword arguments
* `check_point::Bool=true`: whether to also verify that ``p∈\mathcal M` holds, using [`is_point`](@extref ManifoldsBase.is_point)
* `error::Symbol=:none`: if the point is not feasible, this symbol determines how to report the error.
    * `:error`: throws an error
    * `:info`: displays the error message as an @info
    * `:none`: (default) the function just returns true/false
    * `:warn`: displays the error message as a @warning.

The keyword `error=` and all other `kwargs...` are passed on to [`is_point`](@extref ManifoldsBase.is_point) if the point is verfied (see `check_point`).

All other keywords are passed on to `is_poi`
"""
function is_feasible(M, cmo, p; check_point::Bool=true, error::Symbol=:none, kwargs...)
    v = !check_point || is_point(M, p; error=error)
    g = get_inequality_constraint(M, cmo, p, :)
    h = get_equality_constraint(M, cmo, p, :)
    feasible = v && all(g .<= 0) && all(h .== 0)
    # if we are feasible or no error shall be generated
    ((error === :none) || feasible) && return feasible
    # collect information about infeasibily
    if (error === :info) || (error === :warn) || (error === :error)
        s = get_feasibility_status(M, cmo, p; g=g, h=h)
        (error === :error) && throw(ErrorException(s))
        (error === :info) && @info s
        (error === :warn) && @warn s
    end
    return feasible
end

@doc raw"""
    get_feasibility_status(
        M::AbstractManifold,
        cmo::ConstrainedManifoldObjective,
        g = get_inequality_constraints(M, cmo, p),
        h = get_equality_constraints(M, cmo, p),
    )

Generate a message about the feasibiliy of `p` with respect to the [`ConstrainedManifoldObjective`](@ref).
You can also provide the evaluated vectors for the values of `g` and `h` as keyword arguments,
in case you had them evaluated before.
"""
function get_feasibility_status(
    M,
    cmo,
    p;
    g=get_inequality_constraints(M, cmo, p),
    h=get_equality_constraints(M, cmo, p),
)
    g_violated = sum(g .> 0)
    h_violated = sum(h .!= 0)
    return """
    The point $p on $M is not feasible for the provided constants.

    * There are $(g_violated) of $(length(g)) inequality constraints violated. $(
    g_violated > 0 ? "The sum of violation is $(sum(max.(g,Ref(0))))." : ""
    )
    * There are $(h_violated) of $(length(h)) equality constraints violated. $(
    h_violated > 0 ? "The sum of violation is $(sum(abs.(h)))." : ""
    )
    """
end

function Base.show(
    io::IO, ::ConstrainedManifoldObjective{E,V,Eq,IEq}
) where {E<:AbstractEvaluationType,V,Eq,IEq}
    #    return print(io, "ConstrainedManifoldObjective{$E,$V,$Eq,$IEq}.")
    return print(io, "ConstrainedManifoldObjective{$E}")
end
