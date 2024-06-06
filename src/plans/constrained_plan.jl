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
    equality_type=_vector_function_type_hint(h),
    equality_gradient_type=_vector_function_type_hint(grad_h),
    equality_hessian_type=_vector_function_type_hint(hess_h),
    inequality_type=_vector_function_type_hint(g),
    inequality_gradient_type=_vector_function_type_hint(grad_g),
    inequality_hessian_type=_vector_function_type_hint(hess_g),
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
        co::ConstrainedManifoldObjetive;
        range=NestedPowerRepresentation(),
        gradient_equality_range=range,
        gradient_inequality_range=range
        hessian_equality_range=range,
        hessian_inequality_range=range
    )

Creates a constrained manopt problem specifying an [`AbstractPowerRepresentation`](@ref)
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
    grad_ineqality_range::GR
    hess_equality_range::HHR
    hess_ineqality_range::GHR
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
    O,
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
    equality_constraints_length(co::ConstrainedManifoldObjective)

Return the number of equality constraints of an [`ConstrainedManifoldObjective`](@ref).
This acts transparently through [`AbstractDecoratedManifoldObjective`](@ref)s
"""
function equality_constraints_length(co::ConstrainedManifoldObjective)
    return length(co.equality_constraints)
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
        "get_contsraints will be removed in a future release, use `get_equality_constraint($mp, $p, :)` and `get_equality_constraint($mp, $p, :)`, respectively",
        :get_constraints,
    )
    return [
        get_inequality_constraint(get_manifold(mp), get_objective(mp), p, :),
        get_equality_constraint(get_manifold(mp), get_objective(mp), p, :),
    ]
end
function get_constraints(M::AbstractManifold, co::ConstrainedManifoldObjective, p)
    Base.depwarn(
        "get_constraints will be removed in a future release, use `get_equality_constraint($M, $co, $p, :)` and `get_equality_constraint($M, $co, $p, :)`, respectively",
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
    get_equality_constraint(problem, p, j=:)
    get_equality_constraint(manifold, objective, p, j=:)

evaluate the equality constraint of a [`ConstrainedManifoldObjective`](@ref) `objective`.
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
    get_inequality_constraint(amp::AbstractManoptProblem, p, i=:)
    get_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i=:, range=NestedPowerRepresentation())
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
    get_grad_equality_constraint(amp::AbstractManoptProblem, p, i)
    get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i, range=NestedPowerRepresentation())
    get_grad_equality_constraint!(amp::AbstractManoptProblem, X, p, i)
    get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, i, range=NestedPowerRepresentation())

evaluate the gradient or gradients  of the equality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(x)``,

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
    get_grad_inequality_constraint(amp::AbstractManoptProblem, p, i=:)
    get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i=:, range=NestedPowerRepresentation())
    get_grad_inequality_constraint!(amp::AbstractManoptProblem, X, p, i=:)
    get_grad_inequality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, i=:, range=NestedPowerRepresentation())

Evaluate the gradient or gradients of the inequality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(x)``,

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
        get_manifold(cmp), get_objective(cmp), p, j, cmp.grad_ineqality_range
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
        get_manifold(cmp), X, get_objective(cmp), p, j, cmp.grad_ineqality_range
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
    isnothing(co.equality_constraints) && (return X)
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
    get_hess_equality_constraint(amp::AbstractManoptProblem, p, i=:)
    get_hess_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i, range=NestedPowerRepresentation())
    get_hess_equality_constraint!(amp::AbstractManoptProblem, X, p, i=:)
    get_hess_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, i, range=NestedPowerRepresentation())

evaluate the Hessian or Hessians of the equality constraint ``(\operatorname{Hess} h(p))_j`` or ``\operatorname{Hess} h_j(x)``,

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
    get_hess_inequality_constraint(amp::AbstractManoptProblem, p, X, i=:)
    get_hess_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i=:, range=NestedPowerRepresentation())
    get_hess_inequality_constraint!(amp::AbstractManoptProblem, Y, p, i=:)
    get_hess_inequality_constraint!(M::AbstractManifold, Y, co::ConstrainedManifoldObjective, p, X, i=:, range=NestedPowerRepresentation())

evaluate the Hessian or Hessians of the inequality constraint ``(\operatorname{Hess} g(p)[X])_j`` or ``\operatorname{Hess} g_j(x)[X]``,

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
        get_manifold(cmp), get_objective(cmp), p, X, j, cmp.hess_ineqality_range
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
        get_manifold(cmp), Y, get_objective(cmp), p, X, j, cmp.hess_ineqality_range
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
    isnothing(co.equality_constraints) && (return X)
    return get_hessian!(M, Y, co.inequality_constraints, p, X, j, range)
end

@doc raw"""
    inequality_constraints_length(co::ConstrainedManifoldObjective)

Return the number of inequality constraints of an [`ConstrainedManifoldObjective`](@ref).
This acts transparently through [`AbstractDecoratedManifoldObjective`](@ref)s
"""
function inequality_constraints_length(co::ConstrainedManifoldObjective)
    return length(co.inequality_constraints)
end
function inequality_constraints_length(co::AbstractDecoratedManifoldObjective)
    return inequality_constraints_length(get_objective(co, false))
end

function Base.show(
    io::IO, ::ConstrainedManifoldObjective{E,V,Eq,IEq}
) where {E<:AbstractEvaluationType,V,Eq,IEq}
    #    return print(io, "ConstrainedManifoldObjective{$E,$V,$Eq,$IEq}.")
    return print(io, "ConstrainedManifoldObjective{$E}")
end
