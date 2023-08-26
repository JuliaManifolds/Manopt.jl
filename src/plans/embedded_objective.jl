@doc raw"""
    EmbeddedManifoldObjective{P, T, E, O2, O1<:AbstractManifoldObjective{E}} <:
       AbstractDecoratedManifoldObjective{O2, O1}

Declare an objective to be defined in the embedding.
This also declares the gradient to be defined in the embedding,
and especially being the Riesz representer with respect to the metric in the embedding.
The types can be used to still dispatch on also the undecorated objective type `O2`.

# Fields
* `objective` – the objective that is defined in the embedding
* `p`         - (`nothing`) a point in the embedding.
* `X`         - (`nothing`) a tangent vector in the embedding

When a point in the embedding `p` is provided, `embed!` is used in place of this point to reduce
memory allocations. Similarly `X` is used when embedding tangent vectors

"""
struct EmbeddedManifoldObjective{P,T,E,O2,O1<:AbstractManifoldObjective{E}} <:
       AbstractDecoratedManifoldObjective{E,O2}
    objective::O1
    p::P
    X::T
end
function EmbeddedManifoldObjective(
    o::O, p::P=nothing, X::T=nothing
) where {P,T,E<:AbstractEvaluationType,O<:AbstractManifoldObjective{E}}
    return EmbeddedManifoldObjective{P,T,E,O,O}(o, p, X)
end
function EmbeddedManifoldObjective(
    o::O1, p::P=nothing, X::T=nothing
) where {
    P,
    T,
    E<:AbstractEvaluationType,
    O2<:AbstractManifoldObjective,
    O1<:AbstractDecoratedManifoldObjective{E,O2},
}
    return EmbeddedManifoldObjective{P,T,E,P,O}(o, p, X)
end
function EmbeddedManifoldObjective(
    M::AbstractManifold,
    o::O;
    q=rand(M),
    p::P=embed(M, q),
    X::T=embed(M, q, rand(M; vector_at=q)),
) where {P,T,O<:AbstractManifoldObjective}
    return EmbeddedManifoldObjective(o, p, X)
end

function embed!(M::AbstractManifold, ::EmbeddedManifoldObjective{Nothing}, p)
    return embed(M, p)
end
function embed!(M::AbstractManifold, emo::EmbeddedManifoldObjective{P}, p) where {P}
    embed!(M, emo.p, p)
    return emo.p
end
function embed!(M::AbstractManifold, ::EmbeddedManifoldObjective{P,Nothing}, p, X) where {P}
    return embed(M, p, X)
end
function embed!(M::AbstractManifold, emo::EmbeddedManifoldObjective{P,T}, p, X) where {P,T}
    embed!(M, emo.X, p, X)
    return emo.X
end

@doc raw"""
    get_cost(M, emo::EmbeddedManifoldObjective, p)

Evaluate the cost function of an objective defined in the embedding, that is
call [`embed`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.embed-Tuple{AbstractManifold,%20Any})
on the point `p` and call the original cost on this point.

If `emo.p` is not `nothing`, the embedding is done in place of `emo.p`.
"""
function get_cost(M, emo::EmbeddedManifoldObjective, p)
    q = embed!(M, emo, p)
    return get_cost(get_embedding(M), emo.objective, q)
end
function get_cost_function(emo::EmbeddedManifoldObjective)
    return (M, p) -> get_cost(M, emo, p)
end

@doc raw"""
    get_gradient(M, emo::EmbeddedManifoldObjective, p)
    get_gradient!(M, X, emo::EmbeddedManifoldObjective, p)

Evaluate the gradient function of an objective defined in the embedding, that is
call [`embed`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.embed-Tuple{AbstractManifold,%20Any})
on the point `p` and call the original cost on this point.
And convert the gradient using [`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}) on the result.

If `emo.p` is not `nothing`, the embedding is done in place of `emo.p`.
"""
function get_gradient(M, emo::EmbeddedManifoldObjective{P,Nothing}, p) where {P}
    q = embed!(M, emo, p)
    return riemannian_gradient(M, p, get_gradient(get_embedding(M), emo.objective, q))
end
function get_gradient(M, emo::EmbeddedManifoldObjective{P,T}, p) where {P,T}
    q = embed!(M, emo, p)
    get_gradient!(get_embedding(M), emo.X, emo.objective, q)
    return riemannian_gradient(M, p, emo.X)
end
function get_gradient!(M, X, emo::EmbeddedManifoldObjective{P,Nothing}, p) where {P}
    q = embed!(M, emo, p)
    riemannian_gradient!(M, X, p, get_gradient(get_embedding(M), emo.objective, q))
    return X
end
function get_gradient!(M, X, emo::EmbeddedManifoldObjective{Nothing,T}, p) where {T}
    get_gradient!(get_embedding(M), emo.X, emo, embed(M, p))
    riemannian_gradient!(M, X, p, emo.X)
    return X
end
function get_gradient!(M, X, emo::EmbeddedManifoldObjective{P,T}, p) where {P,T}
    q = embed!(M, emo, p)
    get_gradient!(get_embedding(M), emo.X, emo, q)
    riemannian_gradient!(M, X, p, emo.X)
    return X
end
#
# Hessian
#
@doc raw"""
    get_Hessian(M, emo::EmbeddedManifoldObjective, p, X)
    get_Hessian!(M, Y, emo::EmbeddedManifoldObjective, p, X)

Evaluate the Hessian of an objective defined in the embedding, that is
call [`embed`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.embed-Tuple{AbstractManifold,%20Any})
on the point `p` and the direction `X` before passing them to the embedded Hessian,
and use [`riemannian_Hessian`]() on the result to convert it to a Riemannian one.

If `emo.p` and/or `emp.X` are not `nothing`, the embedding and Hessian evaluation is done in place of these variables.
"""
function get_hessian(M, emo::EmbeddedManifoldObjective{P,Nothing}, p, X) where {P}
    q = embed!(M, emo, p)
    return riemannian_Hessian(
        M,
        p,
        get_gradient(get_embedding(M), emo.objective, q),
        get_hessian(get_embedding(M), emo.objective, q, embed(M, p, X)),
        X,
    )
end
function get_hessian(M, emo::EmbeddedManifoldObjective{P,T}, p) where {P,T}
    q = embed!(M, emo, p)
    get_gradient!(get_embedding(M), emo.X, emo.objective, embed(M, p))
    return riemannian_Hessian(
        M, p, emo.X, get_hessian(get_embedding(M), emo.objective, q, embed(M, p, X)), X
    )
end
function get_hessian!(M, Y, emo::EmbeddedManifoldObjective{P,Nothing}, p, X) where {P}
    q = embed!(M, emo, p)
    riemannian_Hessian!(
        M,
        Y,
        p,
        get_gradient(get_embedding(M), emo.objective, q),
        get_hessian(get_embedding(M), emo.objective, q, embed(M, p, X)),
        X,
    )
    return Y
end
function get_hessian!(M, Y, emo::EmbeddedManifoldObjective{Nothing,T}, p) where {T}
    q = embed!(M, emo, p)
    get_gradient!(get_embedding(M), emo.X, emo.objective, embed(M, p))
    riemannian_Hessian!(
        M, Y, p, emo.X, get_hessian(get_embedding(M), emo.objective, q, embed(M, p, X)), X
    )
    return Y
end
#
# Constraints
#
"""
    get_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)

Return the vector ``(g_1(p),...g_m(p),h_1(p),...,h_n(p))`` from the [`ConstrainedManifoldObjective`](@ref) `P`
containing the values of all constraints at `p`, where the original constraint(s) are defined in the embedding.
"""
function get_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = embed!(M, emo, p)
    return [
        get_inequality_constraints(M, emo.objective, q),
        get_equality_constraints(M, emo.objective, q),
    ]
end
@doc raw"""
    get_equality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)

evaluate all equality constraints ``h(p)`` of ``\bigl(h_1(p), h_2(p),\ldots,h_p(p)\bigr)``
of the [`EmbeddedManifoldObjective`](@ref) `emo` at ``p``.
"""
function get_equality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = embed!(M, emo, p)
    return get_equality_constraints(M, emo.objective, q)
end
@doc raw"""
    get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)

evaluate the `j`th equality constraint ``(h(p))_j`` or ``h_j(p)`` for an [`EmbeddedManifoldObjective`](@ref).
"""
function get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    q = embed!(M, emo, p)
    return get_equality_constraint(M, emo.objective, q, j)
end
@doc raw"""
    get_inequality_constraints(M::AbstractManifold, co::EmbeddedManifoldObjective, p)

Evaluate all inequality constraints ``g(p)`` or ``\bigl(g_1(p), g_2(p),\ldots,g_m(p)\bigr)``
of the [`EmbeddedManifoldObjective`](@ref) `emo` at ``p``.
"""
function get_inequality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = embed!(M, emo, p)
    return get_inequality_constraints(M, emo.objective, q)
end
@doc raw"""
    get_inequality_constraint(M::AbstractManifold, co::EmbeddedManifoldObjective, p, i)

evaluate one equality constraint ``(g(p))_i`` or ``g_i(p)`` in the embedding.
"""
function get_inequality_constraint(
    M::AbstractManifold, emo::EmbeddedManifoldObjective, p, i
)
    q = embed!(M, emo, p)
    return get_inequality_constraint(M, emo.objective, q, i)
end

function get_grad_equality_constraint(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,Nothing}, p, j
) where {P}
    q = embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M)emo.objective, q, j)
    return riemannian_gradient(M, p, Z)
end
function get_grad_equality_constraint(
    M::AbstractManifold, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_equality_constraint!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P,Nothing}, p, j
) where {P}
    q = embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M)emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_equality_constraint!(
    M::AbstractManifold, Y, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end

function get_grad_equality_constraints(
    M::AbstractManifold, emo::EmbeddedManifoldObjective, p
)
    q = embed!(M, emo, p)
    Z = get_grad_equality_constraints(get_embedding(M)emo.objective, q)
    return [riemannian_gradient(M, p, Zj) for Zj in Z]
end
function get_grad_equality_constraints!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective, p
)
    q = embed!(M, emo, p)
    Z = get_grad_equality_constraints(get_embedding(M)emo.objective, q)
    for (Yj, Zj) in zip(Y, Z)
        riemannian_gradient!(M, Yj, p, Zj)
    end
    return Y
end

function get_grad_inequality_constraint(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,Nothing}, p, j
) where {P}
    q = embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M)emo.objective, q, j)
    return riemannian_gradient(M, p, Z)
end
function get_grad_inequality_constraint(
    M::AbstractManifold, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_inequality_constraint!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P,Nothing}, p, j
) where {P}
    q = embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M)emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_inequality_constraint!(
    M::AbstractManifold, Y, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end

function get_grad_inequality_constraints(
    M::AbstractManifold, emo::EmbeddedManifoldObjective, p
)
    q = embed!(M, emo, p)
    Z = get_grad_inequality_constraints(get_embedding(M)emo.objective, q)
    return [riemannian_gradient(M, p, Zj) for Zj in Z]
end
function get_grad_inequality_constraints!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective, p
)
    q = embed!(M, emo, p)
    Z = get_grad_inequality_constraints(get_embedding(M)emo.objective, q)
    for (Yj, Zj) in zip(Y, Z)
        riemannian_gradient!(M, Yj, p, Zj)
    end
    return Y
end
