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
    o::O, p::P=missing, X::T=missing
) where {P,T,E<:AbstractEvaluationType,O<:AbstractManifoldObjective{E}}
    return EmbeddedManifoldObjective{P,T,E,O,O}(o, p, X)
end
function EmbeddedManifoldObjective(
    o::O1, p::P=missing, X::T=missing
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

# dispatch whether to do this in place or not
function local_embed!(M::AbstractManifold, ::EmbeddedManifoldObjective{Missing}, p)
    return embed(M, p)
end
function local_embed!(M::AbstractManifold, emo::EmbeddedManifoldObjective{P}, p) where {P}
    embed!(M, emo.p, p)
    return emo.p
end
function local_embed!(
    M::AbstractManifold, ::EmbeddedManifoldObjective{P,Missing}, p, X
) where {P}
    return embed(M, p, X)
end
function local_embed!(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,T}, p, X
) where {P,T}
    embed!(M, emo.X, p, X)
    return emo.X
end

@doc raw"""
    get_cost(M::AbstractManifold,emo::EmbeddedManifoldObjective, p)

Evaluate the cost function of an objective defined in the embedding, i.e. embed `p`
before calling the cost function stored in the [`EmbeddedManifoldObjective`](@ref Manopt.EmbeddedManifoldObjective).
"""
function get_cost(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = local_embed!(M, emo, p)
    return get_cost(get_embedding(M), emo.objective, q)
end
function get_cost_function(emo::EmbeddedManifoldObjective)
    return (M, p) -> get_cost(M, emo, p)
end

@doc raw"""
    get_gradient(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    get_gradient!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p)

Evaluate the gradient function of an objective defined in the embedding, that is embed `p`
before calling the gradient function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_gradient(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,Missing}, p
) where {P}
    q = local_embed!(M, emo, p)
    return riemannian_gradient(M, p, get_gradient(get_embedding(M), emo.objective, q))
end
function get_gradient(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,T}, p
) where {P,T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M), emo.X, emo.objective, q)
    return riemannian_gradient(M, p, emo.X)
end
function get_gradient!(
    M::AbstractManifold, X, emo::EmbeddedManifoldObjective{P,Missing}, p
) where {P}
    q = local_embed!(M, emo, p)
    riemannian_gradient!(M, X, p, get_gradient(get_embedding(M), emo.objective, q))
    return X
end
function get_gradient!(
    M::AbstractManifold, X, emo::EmbeddedManifoldObjective{P,T}, p
) where {P,T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M), emo.X, emo.objective, q)
    riemannian_gradient!(M, X, p, emo.X)
    return X
end
#
# Hessian
#
@doc raw"""
    get_hessian(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, X)
    get_hessian!(M::AbstractManifold, Y, emo::EmbeddedManifoldObjective, p, X)

Evaluate the Hessian of an objective defined in the embedding, that is embed `p` and `X`
before calling the Hessiand function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned Hessian is then converted to a Riemannian Hessian calling
 [`riemannian_Hessian`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library/#ManifoldDiff.riemannian_Hessian-Tuple{AbstractManifold,%20Any,%20Any,%20Any,%20Any}).
"""
function get_hessian(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,Missing}, p, X
) where {P}
    q = local_embed!(M, emo, p)
    return riemannian_Hessian(
        M,
        p,
        get_gradient(get_embedding(M), emo.objective, q),
        get_hessian(get_embedding(M), emo.objective, q, embed(M, p, X)),
        X,
    )
end
function get_hessian(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,T}, p
) where {P,T}
    q = local_embed!(M, emo, p)
    get_gradient!(get_embedding(M), emo.X, emo.objective, embed(M, p))
    return riemannian_Hessian(
        M, p, emo.X, get_hessian(get_embedding(M), emo.objective, q, embed(M, p, X)), X
    )
end
function get_hessian!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P,Missing}, p, X
) where {P}
    q = local_embed!(M, emo, p)
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
function get_hessian!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P,T}, p, X
) where {P,T}
    q = local_embed!(M, emo, p)
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

Return the vector ``(g_1(p),...g_m(p),h_1(p),...,h_n(p))`` defined in the embedding, that is embed `p`
before calling the constraint function(s) stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = local_embed!(M, emo, p)
    return [
        get_inequality_constraints(M, emo.objective, q),
        get_equality_constraints(M, emo.objective, q),
    ]
end
@doc raw"""
    get_equality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)

Evaluate all equality constraints ``h(p)`` of ``\bigl(h_1(p), h_2(p),\ldots,h_p(p)\bigr)``
defined in the embedding, that is embed `p`
before calling the constraint function(s) stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_equality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = local_embed!(M, emo, p)
    return get_equality_constraints(M, emo.objective, q)
end
@doc raw"""
    get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)

evaluate the `j`s equality constraint ``h_j(p)`` defined in the embedding, that is embed `p`
before calling the constraint function(s) stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    q = local_embed!(M, emo, p)
    return get_equality_constraint(M, emo.objective, q, j)
end
@doc raw"""
    get_inequality_constraints(M::AbstractManifold, ems::EmbeddedManifoldObjective, p)

Evaluate all inequality constraints ``g(p)`` of ``\bigl(g_1(p), g_2(p),\ldots,g_m(p)\bigr)``
defined in the embedding, that is embed `p`
before calling the constraint function(s) stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_inequality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    q = local_embed!(M, emo, p)
    return get_inequality_constraints(M, emo.objective, q)
end

@doc raw"""
    get_inequality_constraint(M::AbstractManifold, ems::EmbeddedManifoldObjective, p, i)

Evaluate the `i`s inequality constraint ``g_i(p)`` defined in the embedding, that is embed `p`
before calling the constraint function(s) stored in the [`EmbeddedManifoldObjective`](@ref).
"""
function get_inequality_constraint(
    M::AbstractManifold, emo::EmbeddedManifoldObjective, p, i
)
    q = local_embed!(M, emo, p)
    return get_inequality_constraint(M, emo.objective, q, i)
end
@doc raw"""
    X = get_grad_equality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, j)
    get_grad_equality_constraint!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p, j)

evaluate the gradient of the `j`th equality constraint ``\operatorname{grad} h_j(p)`` defined in the embedding, that is embed `p`
before calling the gradient function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_equality_constraint(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,Missing}, p, j
) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M)emo.objective, q, j)
    return riemannian_gradient(M, p, Z)
end
function get_grad_equality_constraint(
    M::AbstractManifold, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = local_embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_equality_constraint!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P,Missing}, p, j
) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraint(get_embedding(M)emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_equality_constraint!(
    M::AbstractManifold, Y, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = local_embed!(M, emo, p)
    get_grad_equality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
@doc raw"""
    X = get_grad_equality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    get_grad_equality_constraints!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p)

evaluate the gradients of the the equality constraints ``\operatorname{grad} h(p)`` defined in the embedding, that is embed `p`
before calling the gradient function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned gradients are then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_equality_constraints(
    M::AbstractManifold, emo::EmbeddedManifoldObjective, p
)
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraints(get_embedding(M)emo.objective, q)
    return [riemannian_gradient(M, p, Zj) for Zj in Z]
end
function get_grad_equality_constraints!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective, p
)
    q = local_embed!(M, emo, p)
    Z = get_grad_equality_constraints(get_embedding(M)emo.objective, q)
    for (Yj, Zj) in zip(Y, Z)
        riemannian_gradient!(M, Yj, p, Zj)
    end
    return Y
end
@doc raw"""
    X = get_grad_inequality_constraint(M::AbstractManifold, emo::EmbeddedManifoldObjective, p, i)
    get_grad_inequality_constraint!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p, i)

evaluate the gradient of the `i`th inequality constraint ``\operatorname{grad} g_i(p)`` defined in the embedding, that is embed `p`
before calling the gradient function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned gradient is then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_inequality_constraint(
    M::AbstractManifold, emo::EmbeddedManifoldObjective{P,Missing}, p, j
) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M)emo.objective, q, j)
    return riemannian_gradient(M, p, Z)
end
function get_grad_inequality_constraint(
    M::AbstractManifold, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = local_embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    return riemannian_gradient(M, p, emo.X)
end
function get_grad_inequality_constraint!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective{P,Missing}, p, j
) where {P}
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraint(get_embedding(M), emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
function get_grad_inequality_constraint!(
    M::AbstractManifold, Y, emo::AbstractDecoratedManifoldObjective{P,T}, p, j
) where {P,T}
    q = local_embed!(M, emo, p)
    get_grad_inequality_constraint!(get_embedding(M), emo.X, emo.objective, q, j)
    riemannian_gradient!(M, Y, p, Z)
    return Y
end
@doc raw"""
    X = get_grad_inequality_constraints(M::AbstractManifold, emo::EmbeddedManifoldObjective, p)
    get_grad_inequality_constraints!(M::AbstractManifold, X, emo::EmbeddedManifoldObjective, p)

evaluate the gradients of the the inequality constraints ``\operatorname{grad} g(p)`` defined in the embedding, that is embed `p`
before calling the gradient function stored in the [`EmbeddedManifoldObjective`](@ref).

The returned gradients are then converted to a Riemannian gradient calling
[`riemannian_gradient`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.riemannian_gradient-Tuple{AbstractManifold,%20Any,%20Any}).
"""
function get_grad_inequality_constraints(
    M::AbstractManifold, emo::EmbeddedManifoldObjective, p
)
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraints(get_embedding(M), emo.objective, q)
    return [riemannian_gradient(M, p, Zj) for Zj in Z]
end
function get_grad_inequality_constraints!(
    M::AbstractManifold, Y, emo::EmbeddedManifoldObjective, p
)
    q = local_embed!(M, emo, p)
    Z = get_grad_inequality_constraints(get_embedding(M), emo.objective, q)
    for (Yj, Zj) in zip(Y, Z)
        riemannian_gradient!(M, Yj, p, Zj)
    end
    return Y
end

function show(io::IO, emo::EmbeddedManifoldObjective{P,T}) where {P,T}
    return print(io, "EmbeddedManifoldObjective{$P,$T} of an $(emo.objective)")
end
