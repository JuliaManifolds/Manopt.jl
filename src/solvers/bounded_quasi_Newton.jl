
struct QuasiNewtonBoundedDirectionUpdate{
    UA<:AbstractQuasiNewtonDirectionUpdate,UP<:AbstractQuasiNewtonDirectionUpdate,TX
} <: AbstractQuasiNewtonDirectionUpdate
    update_active::UA
    update_passive::UP
    grad_active::TX
    grad_passive::TX
    dir_active::TX
    dir_passive::TX
end

function QuasiNewtonBoundedDirectionUpdate(
    M::AbstractManifold,
    p,
    update_active::AbstractQuasiNewtonDirectionUpdate,
    update_passive::AbstractQuasiNewtonDirectionUpdate;
    initial_vector::T=zero_vector(M, p),
) where {T}
    return QuasiNewtonBoundedDirectionUpdate{typeof(update_active),typeof(update_passive),T}(
        update_active,
        update_passive,
        copy(M, initial_vector),
        copy(M, initial_vector),
        copy(M, initial_vector),
        copy(M, initial_vector),
    )
end
function (d::QuasiNewtonBoundedDirectionUpdate)(mp, st)
    r = zero_vector(get_manifold(mp), get_iterate(st))
    return d(r, mp, st)
end
function (d::QuasiNewtonBoundedDirectionUpdate)(r, mp, st)
    M = get_manifold(mp)
    p = get_iterate(st)
    grad = get_gradient(st)
    copyto!(M, r, p, grad)
    project_active!(M, d.grad_active, p, r)
    d.grad_passive .= r .- d.grad_active

    apply_to_vector!(d.dir_active, d.update_active, mp, st, d.grad_active)
    apply_to_vector!(d.dir_passive, d.update_passive, mp, st, d.grad_passive)

    r .= d.dir_active .+ d.dir_passive
    if inner(M, p, r, grad) >= 0
        r .= (-1) .* grad
        initialize_update!(d.update_active)
        initialize_update!(d.update_passive)
    end
    return r
end

@doc raw"""
    get_active_constraints(M::AbstractManifold, p, X_e; eps::Real=eps(number_eltype(p)))

Return `k`-element vector where ``i``th element is true when ``g_i(p) \in [0, \text{eps})``
and ``D g_i(p)[X_e] < 0``, and false otherwise. These conditions directly generalize
Eq. (32) from [Bertsekas:1982].

The manifold with corners `M` is defined as a subset of ``\mathbb{R}^n`` such that
``g_i(x) \geq 0`` for ``i \in 1, 2, \dots, k``.
"""
function get_active_constraints(
    M::AbstractManifold, p, X_e; eps::Real=eps(number_eltype(p))
) end

function initialize_update!(s::QuasiNewtonBoundedDirectionUpdate)
    initialize_update!(s.update_active)
    initialize_update!(s.update_passive)
    return s
end

function project_active!(::AbstractManifold, Y, p, X) end
