@doc """
    EmbeddedManifoldObjective{P, T, E, O2, O1<:AbstractManifoldObjective{E}} <:
       AbstractDecoratedManifoldObjective{E,O2}

Declare an objective to be defined in the embedding.
This also declares the gradient to be defined in the embedding,
and especially being the Riesz representer with respect to the metric in the embedding.
The types can be used to still dispatch on also the undecorated objective type `O2`.

# Fields

* `objective`: the objective that is defined in the embedding
* `p=nothing`: a point in the embedding.
* `X=nothing`: a tangent vector in the embedding

When a point in the embedding `p` is provided, `embed!` is used in place of this point to reduce
memory allocations. Similarly `X` is used when embedding tangent vectors
"""
struct EmbeddedManifoldObjective{P, T, E, O2, O1 <: AbstractManifoldObjective{E}} <:
    AbstractDecoratedManifoldObjective{E, O2}
    objective::O1
    p::P
    X::T
end
function EmbeddedManifoldObjective(
        o::O, p::P = missing, X::T = missing
    ) where {P, T, E <: AbstractEvaluationType, O <: AbstractManifoldObjective{E}}
    return EmbeddedManifoldObjective{P, T, E, O, O}(o, p, X)
end
function EmbeddedManifoldObjective(
        o::O1, p::P = missing, X::T = missing
    ) where {
        P,
        T,
        E <: AbstractEvaluationType,
        O2 <: AbstractManifoldObjective,
        O1 <: AbstractDecoratedManifoldObjective{E, O2},
    }
    return EmbeddedManifoldObjective{P, T, E, O2, O1}(o, p, X)
end
function EmbeddedManifoldObjective(
        M::AbstractManifold,
        o::O;
        q = rand(M),
        p::P = embed(M, q),
        X::T = embed(M, q, rand(M; vector_at = q)),
    ) where {P, T, O <: AbstractManifoldObjective}
    return EmbeddedManifoldObjective(o, p, X)
end

# dispatch whether to do this in place or not
function local_embed!(M::AbstractManifold, ::EmbeddedManifoldObjective{Missing}, p)
    return embed(M, p)
end
function local_embed!(M::AbstractManifold, emo::EmbeddedManifoldObjective{P}, p) where {P}
    embed!(M, emo.p, p)
    return emo.p
end

@doc """
    get_cost(M::AbstractManifold,emo::EmbeddedManifoldObjective, p)

Evaluate the cost function of an objective defined in the embedding by first embedding `p`
before calling the cost function stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_cost(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = local_embed!(M, emo, p)
    return get_cost(get_embedding(M, typeof(p)), emo.objective, q)
end

function get_cost_function(emo::EmbeddedManifoldObjective, recursive = false)
    recursive && (return get_cost_function(emo.objective, recursive))
    return (M, p) -> get_cost(M, emo, p)
end
@doc """
    get_gradient(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    get_gradient!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p)

Evaluate the gradient function of an objective defined in the embedding, that is embed `p`
before calling the gradient function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_gradient(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p
    ) where {P}
    q = local_embed!(M, emo, p)
    return riemannian_gradient(M, p, get_gradient(get_embedding(M, typeof(p)), emo.objective, q))
end
function get_gradient(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, q)
    return riemannian_gradient(M, p, emo.X)
end
function get_gradient!(
        M::AbstractManifold, X, emo::EmbeddedManifoldObjective{P, Missing}, p
    ) where {P}
    q = local_embed!(M, emo, p)
    riemannian_gradient!(M, X, p, get_gradient(get_embedding(M, typeof(p)), emo.objective, q))
    return X
end
function get_gradient!(
        M::AbstractManifold, X, emo::EmbeddedManifoldObjective{P, T}, p
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, q)
    riemannian_gradient!(M, X, p, emo.X)
    return X
end

function get_gradient_function(
        emo::EmbeddedManifoldObjective{P, T, AllocatingEvaluation}, recursive = false
    ) where {P, T}
    recursive && (return get_gradient_function(emo.objective, recursive))
    return (M, p) -> get_gradient(M, emo, p)
end
function get_gradient_function(
        emo::EmbeddedManifoldObjective{P, T, InplaceEvaluation}, recursive = false
    ) where {P, T}
    recursive && (return get_gradient_function(emo.objective, recursive))
    return (M, X, p) -> get_gradient!(M, X, emo, p)
end
#
# Hessian
#
@doc """
    get_hessian(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, X)
    get_hessian!(M::AbstractManifold, Y, emo::EmbeddedManifoldObjective, p, X)

Evaluate the Hessian of an objective defined in the embedding, that is embed `p` and `X`
before calling the Hessian function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned Hessian is then converted to a Riemannian Hessian calling
 [`riemannian_Hessian`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library/#ManifoldDiff.riemannian_Hessian-Tuple{AbstractManifold,%20Any,%20Any,%20Any,%20Any}).
"""
function get_hessian(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, X
    ) where {P}
    q = local_embed!(M, emo, p)
    return riemannian_Hessian(
        M,
        p,
        get_gradient(get_embedding(M, typeof(p)), emo.objective, q),
        get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)),
        X,
    )
end
function get_hessian(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, X
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, embed(M, p))
    return riemannian_Hessian(
        M, p, emo.X, get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)), X
    )
end
function get_hessian!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, X
    ) where {P}
    q = local_embed!(M, emo, p)
    riemannian_Hessian!(
        M,
        Y,
        p,
        get_gradient(get_embedding(M, typeof(p)), emo.objective, q),
        get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)),
        X,
    )
    return Y
end
function get_hessian!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, X
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M, typeof(p)), emo.X, emo.objective, embed(M, p))
    riemannian_Hessian!(
        M, Y, p, emo.X, get_hessian(get_embedding(M, typeof(p)), emo.objective, q, embed(M, p, X)), X
    )
    return Y
end

function get_hessian_function(
        emo::EmbeddedManifoldObjective{P, T, AllocatingEvaluation}, recursive::Bool = false
    ) where {P, T}
    recursive && (return get_hessian_function(emo.objective, recursive))
    return (M, p, X) -> get_hessian(M, emo, p, X)
end
function get_hessian_function(
        emo::EmbeddedManifoldObjective{P, T, InplaceEvaluation}, recursive::Bool = false
    ) where {P, T}
    recursive && (return get_hessian_function(emo.objective, recursive))
    return (M, Y, p, X) -> get_hessian!(M, Y, emo, p, X)
end

#
# Constraints
#

function get_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = local_embed!(M, emo, p)
    return [
        get_inequality_constraint(M, emo.objective, q, :),
        get_equality_constraint(M, emo.objective, q, :),
    ]
end
@doc """
    get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)

evaluate the `j`s equality constraint ``h_j(p)`` defined in the embedding, that is embed `p`
before calling the constraint functions stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    q = local_embed!(M, emo, p)
    return get_equality_constraint(M, emo.objective, q, j)
end
@doc """
    get_inequality_constraint(M::AbstractManifold, ems::EmbeddedManifoldObjective, p, i)

Evaluate the `i`s inequality constraint ``g_i(p)`` defined in the embedding, that is embed `p`
before calling the constraint functions stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective, p, i
    )
    q = local_embed!(M, emo, p)
    return get_inequality_constraint(M, emo.objective, q, i)
end
@doc """
    X = get_grad_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    get_grad_equality_constraint!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p, j)

Evaluate the gradient of the `j`th equality constraint ``$(_tex(:grad)) h_j(p)``
defined in the embedding, that is embed `p` before calling the gradient function stored in
the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, j::Integer
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return riemannian_gradient(M, p, Z)
end
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, j
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return [riemannian_gradient(M, p, X) for X in Z]
end
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, j::Integer
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, j)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_equality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, j
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Xs = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Ys = [riemannian_gradient(M, p, X) for X in Xs]
    return Ys
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, j::Integer
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, j
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, j::Integer
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, j)
    riemannian_gradient!(M, Y, p, emo.X)
    return Y
end
function get_grad_equality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, j
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
@doc """
    X = get_grad_inequality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    get_grad_inequality_constraint!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p, j)

Evaluate the gradient of the `j`th inequality constraint ``$(_tex(:grad)) g_j(p)``
defined in the embedding, that is embed `p` before calling the gradient function stored in
the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, i::Integer
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, i)
    return riemannian_gradient(M, p, Z)
end
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, Missing}, p, j
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return [riemannian_gradient(M, p, X) for X in Z]
end
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, i::Integer
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, i)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_inequality_constraint(
        M::AbstractManifold, emo::EmbeddedManifoldObjective{P, T}, p, j
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    return [riemannian_gradient(M, p, X) for X in Z]
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, i::Integer
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, i)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, Missing}, p, j
    ) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, i::Integer
    ) where {P, T}
    q = local_embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M, typeof(p)), emo.X, emo.objective, q, i)
    riemannian_gradient!(M, Y, p, emo.X)
    return Y
end
function get_grad_inequality_constraint!(
        M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P, T}, p, j
    ) where {P, T}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M, typeof(p)), emo.objective, q, j)
    Y .= [riemannian_gradient(M, p, X) for X in Z]
    return Y
end
function show(io::IO, emo::EmbeddedManifoldObjective)
    return print(io, "EmbeddedManifoldObjective($(emo.objective), $(emo.p), $(emo.X))")
end
function status_summary(io::IO, emo::EmbeddedManifoldObjective; context = :default)
    return print(io, status_summary(emo; context = context))
end
function status_summary(emo::EmbeddedManifoldObjective{P, T}; context = :default) where {P, T}
    _is_inline(context) && return "An embedded objective of $(status_summary(emo.objective; context = context))"
    p_str = !(ismissing(emo.p)) ? "* for a point of type $P" : ""
    X_str = !(ismissing(emo.X)) ? "* for a tangent vector of type $T" : ""
    pX_str = (length(p_str) + length(X_str) > 0) ? "\n\n## Temporary memory (in the embedding)\n$(p_str)$(length(p_str) > 0 ? "\n" : "")$(X_str)" : ""
    return """
    An embedded objective

    ## Objective
    $(_MANOPT_INDENT)$(replace(status_summary(emo.objective, context = context), "\n#" => "\n$(_MANOPT_INDENT)##", "\n" => "\n$(_MANOPT_INDENT)"))$(pX_str)"""
end
