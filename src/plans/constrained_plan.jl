@doc raw"""
    ConstrainedManifoldObjective{T<:AbstractEvaluationType, C <: ConstraintType} <: AbstractManifoldObjective{T}

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
        evaluation=AllocatingEvaluation(),
    )

    ConstrainedManifoldObjective(M::AbstractManifold, mho::AbstractManifoldObjective;
        equality_constraints::Union{Nothing, AbstractManifoldObjective} = nothing,
        inequality_constraints::Union{Nothing, AbstractManifoldObjective} = nothing
    ) where {IMO <: Union{Nothing, AbstractManifoldObjective} EMO <: Union{Nothing, AbstractManifoldObjective}}

TODO: Describe constructors
"""
struct ConstrainedManifoldObjective{
    T<:AbstractEvaluationType,
    MO<:AbstractManifoldObjective,
    IMO<:Union{Nothing,AbstractManifoldObjective},
    EMO<:Union{Nothing,AbstractManifoldObjective},
} <: AbstractManifoldObjective{T}
    objective::MO
    inequality_constrinats::IMO
    equality_constraints::EMO
end
function _vector_function_type_hint(f)
    if isnothing(f) && isa(f, AbstractVector)
        return ComponentVectorialType()
    else
        return FunctionVectorialType()
    end
end

function ConstrainedManifoldObjective(
    f,
    grad_f,
    g,
    grad_g,
    h,
    grad_h;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    equality_type=_vector_function_type_hint(g),
    equality_gradient_type=_vector_function_type_hint(grad_g),
    inequality_type=_vector_function_type_hint(h),
    inequality_gradient_type=_vector_function_type_hint(grad_h),
    equality_constraints=-1,
    inequality_constraints=-1,
    kwargs...,
)
    objective = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    if isnothing(g) || isnothing(grad_g)
        eq = nothing
    else
        if equality_constraints < 0
            isa(equality_type, ComponentVectorialType) && (equality_constraints = length(g))
            isa(equality_gradient_type, ComponentVectorialType) &&
                (equality_constraints = length(grad_g))
            # if it is still < 0, this can not be used
            (equality_constraints < 0) && error(
                "Please specify a positive number number of `equality_constraints` (provided $(equality_constraints))",
            )
        end
        eq = VectorGradientFunction(
            g,
            grad_g,
            equality_constraints;
            evaluation=evaluation,
            function_type=equality_type,
            jacobian_type=equality_gradient_type,
        )
    end
    if isnothing(h) || isnothing(grad_h)
        ineq = nothing
    else
        if inequality_constraints < 0
            isa(inequality_type, ComponentVectorialType) &&
                (inequality_constraints = length(g))
            isa(inequality_gradient_type, ComponentVectorialType) &&
                (inequality_constraints = length(grad_g))
            # if it is still < 0, this can not be used
            (inequality_constraints < 0) && error(
                "Please specify a positive number number of `inequality_constraints` (provided $(inequality_constraints))",
            )
        end
        ineq = VectorGradientFunction(
            h,
            grad_h,
            inequality_constraints;
            evaluation=evaluation,
            function_type=inequality_type,
            jacobian_type=inequality_gradient_type,
        )
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
) where {
    E<:AbstractEvaluationType,
    MO<:AbstractManifoldObjective{E},
    IMO<:AbstractManifoldObjective{E},
    EMO<:AbstractManifoldObjective{E},
}
    if isnothing(equality_constraints) && isnothing(inequality_constraints)
        @warn """
        Neither the inequality and the equality constraints are provided.
        Consider calling `get_objective()` on this constraint objective
        and only work on the unconstrained objective instead.
        """
    end
    return ConstrainedManifoldObjective{E,ACT,MO,IMO,EMO}(
        objective, equality_constraints, inequality_constraints; constraint=constraint_type
    )
end

@doc raw"""
    ConstrainedProblem{
        TM <: AbstractManifold,
        O <: AbstractManifoldObjective
        GR <: AbstractManifold
        HR <: AbstractManifold
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
at ``P = (p,…,p) \in \mathcal M^m``, so in the tangent space to the [`PowerManifold`](@extref) ``\mathcal M^m``.
The case where this is a [`NestedPowerRepresentation`](@extref) this agrees with the
interpretation from before, but on power manifolds, more efficient representations exist.

To then access the elements, the range has to be specified. That is what this
problem is for.
"""
struct ConstrainedManoptProblem{
    TM<:AbstractManifold,
    O<:AbstractManifoldObjective,
    GR<:Union{AbstractPowerRepresentation,Nothing},
    HR<:Union{AbstractPowerRepresentation,Nothing},
} <: AbstractManoptProblem{TM}
    manifold::TM
    grad_equality_range::GR
    grad_ineqality_range::HR
    objective::O
end

function ConstrainedManoptProblem(
    M::TM,
    objective::O;
    range=nothing,
    gradient_equality_range::GR=range,
    gradient_inequality_range::HR=range,
) where {
    TM<:AbstractManifold,
    O,
    GR<:Union{AbstractPowerRepresentation,Nothing},
    HR<:Union{AbstractPowerRepresentation,Nothing},
}
    return ConstrainedManifoldObjective{TM,o,GR,HR}(
        M, gradient_equality_range, gradient_inequality_range, objective
    )
end

get_objective(co::ConstrainedManifoldObjective) = co.objective

function get_constraints(mp::AbstractManoptProblem, p)
    Base.depwarn(
        "get_contsraints will be removed in a future release, use `get_equality_constraint($mp, $p, :)` and `get_equality_constraint($mp, $p, :)`, respectively",
        get_constraints,
    )
    return [
        get_inequality_constraint(get_manifold(mp), get_objective(mp), p, :),
        get_equality_constraint(get_manifold(mp), get_objective(mp), p, :),
    ]
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

Base.@deprecate get_equality_constraints(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p
) get_equality_constraint(M, co, p, :)

@doc raw"""
    get_equality_constraint(problem, p, j)
    get_equality_constraint(manifold, objective, p, j, range=nothing)

evaluate the equality constraint of a [`ConstrainedManifoldObjective`](@ref) `objective`.
"""
function get_equality_constraint end

get_equality_constraint(mp::AbstractManoptProblem, p, j) = get_equality_constraint(mp, p, j)
function get_equality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p, j
)
    return get_cost(M, co.equality_constraints, p, j)
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
    M::AbstractManifold, co::ConstrainedManifoldObjective, p
) get_inequality_constraint(M, co, p, :)

@doc raw"""
    get_inequality_constraint(amp::AbstractManoptProblem, p, i)
    get_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i, range=nothing)
"""
function get_inequality_constraint end

function get_inequality_constraint(mp::AbstractManoptProblem, p, j)
    return get_equality_constraint(mp, p, j)
end
function get_inequality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p, j
)
    return get_cost(M, co.inequality_constraints, p, j)
end

@doc raw"""
    get_grad_equality_constraint(amp::AbstractManoptProblem, p, i)
    get_grad_equality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i, range=nothing)
    get_grad_equality_constraint!(amp::AbstractManoptProblem, X, p, i)
    get_grad_equality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, i, range=nothing)

evaluate the gradient or gradients  of the equality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(x)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""
function get_grad_equality_constraint end

function get_grad_equality_constraint(amp::AbstractManoptProblem, p, j)
    return get_grad_equality_constraint(get_manifold(amp), get_objective(amp), p, j)
end
function get_grad_equality_constraint(cmp::ConstrainedManoptProblem, p, j)
    return get_grad_equality_constraint(
        get_manifold(cmp), get_objective(cmp), p, j, cmp.grad_equality_range
    )
end
function get_grad_equality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p, j
)
    return get_gradient(M, co.equality_constraints, p, j, range)
end

function get_grad_equality_constraint!(amp::AbstractManoptProblem, X, p, j)
    return get_grad_equality_constraint!(get_manifold(amp), X, get_objective(amp), p, j)
end
function get_grad_equality_constraint!(cmp::ConstrainedManoptProblem, X, p, j)
    return get_grad_equality_constraint!(
        get_manifold(cmp), X, get_objective(cmp), p, j, cmp.grad_equality_range
    )
end
function get_grad_equality_constraint!(
    M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j
)
    return get_gradient!(M, X, co.equality_constraints, p, j, range)
end

#Depreacte plurals
Base.@deprecate get_grad_equality_constraints(mp::AbstractManoptProblem, p) get_grad_equality_constraint(
    mp, p, :,
)
Base.@deprecate get_grad_equality_constraints(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p
) get_grad_equality_constraint(M, co, p, :)
Base.@deprecate get_grad_equality_constraints!(mp::AbstractManoptProblem, X, p) get_grad_equality_constraint!(
    mp, X, p, :,
)
Base.@deprecate get_grad_equality_constraints!(
    M::AbstractManifold, X, co::ConstrainedManifoldObjective, p
) get_grad_equality_constraint!(M, X, co, p, :)

@doc raw"""
    get_grad_inequality_constraint(amp::AbstractManoptProblem, p, i)
    get_grad_inequality_constraint(M::AbstractManifold, co::ConstrainedManifoldObjective, p, i, range=nothing)
    get_grad_inequality_constraint!(amp::AbstractManoptProblem, X, p, i)
    get_grad_inequality_constraint!(M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, i, range=nothing)

evaluate the gradient or gradients  of the equality constraint ``(\operatorname{grad} h(p))_j`` or ``\operatorname{grad} h_j(x)``,

See also the [`ConstrainedManoptProblem`](@ref) to specify the range of the gradient.
"""
function get_grad_inequality_constraint end

function get_grad_inequality_constraint(amp::AbstractManoptProblem, p, j)
    return get_grad_inequality_constraint(get_manifold(amp), get_objective(amp), p, j)
end
function get_grad_inequality_constraint(cmp::ConstrainedManoptProblem, p, j)
    return get_grad_inequality_constraint(
        get_manifold(cmp), get_objective(cmp), p, j, cmp.grad_inequality_range
    )
end
function get_grad_inequality_constraint(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p, j
)
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
    M::AbstractManifold, X, co::ConstrainedManifoldObjective, p, j
)
    return get_gradient!(M, X, co.inequality_constraints, p, j, range)
end

#Deprecate plurals
Base.@deprecate get_grad_inequality_constraints(mp::AbstractManoptProblem, p) get_grad_inequality_constraint(
    mp, p, :,
)
Base.@deprecate get_grad_inequality_constraints(
    M::AbstractManifold, co::ConstrainedManifoldObjective, p
) get_grad_inequality_constraint(M, co, p, :)
Base.@deprecate get_grad_inequality_constraints!(mp::AbstractManoptProblem, X, p) get_grad_inequality_constraint!(
    mp, X, p, :,
)
Base.@deprecate get_grad_inequality_constraints!(
    M::AbstractManifold, X, co::ConstrainedManifoldObjective, p
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

function Base.show(
    io::IO, ::ConstrainedManifoldObjective{E,V}
) where {E<:AbstractEvaluationType,V}
    return print(io, "ConstrainedManifoldObjective{$E,$V}.")
end
