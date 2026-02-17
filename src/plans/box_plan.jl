"""
    has_anisotropic_max_stepsize(M::AbstractManifold)

Return `true` if `M` has `max_stepsize` that depends on the direction.
For example, if `M` is a `Hyperrectangle`-like manifold with corners, or a product of it
with a standard manifold. Otherwise return `false`.
"""
has_anisotropic_max_stepsize(::AbstractManifold) = false
has_anisotropic_max_stepsize(M::ProductManifold) = any(has_anisotropic_max_stepsize, M.manifolds)

@doc raw"""
    mutable struct LimitedMemoryHessianApproximation end

An approximation of Hessian of a scalar function of the form ``B_0 = θ I``,
``B_{k+1} = B_k - W_k M_k W_k^{\mathrm{T}}``,
where ``\theta > 0`` is an initial scaling guess.
Matrix ``M_k = \left(\begin{smallmatrix}M_{11} & M_{21}^{\mathrm{T}}\\ M_{21} & M_{22}\end{smallmatrix}\right)``
is stored using its blocks.
Blocks ``W_k`` are (implicitly) composed from `memory_y` and `memory_s` stored in `qn_du`
of type [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref).

Initial scale ``\theta`` is stored in the field `initial_scale` but if the memory isn't empty,
the current scale is set to squared norm of $s_k$ divided by inner product of ``s_k`` and ``y_k``
where ``k`` is the oldest index for which the denominator is not equal to 0.

`last_gcd_result` stores the result of the last generalized Cauchy direction search.

See [ByrdNocedalSchnabel:1994](@cite) for details.
"""
mutable struct QuasiNewtonLimitedMemoryBoxDirectionUpdate{
        TDU <: QuasiNewtonLimitedMemoryDirectionUpdate,
        F <: Real,
        T_HM <: AbstractMatrix,
        V <: AbstractVector,
    } <: AbstractQuasiNewtonDirectionUpdate
    # this approximates inverse Hessian
    qn_du::TDU

    # fields for approximating the Hessian
    current_scale::F
    M_11::T_HM
    M_21::T_HM
    M_22::T_HM
    # buffer for calculating W_k blocks
    buffer_inner_Sk_X::V
    buffer_inner_Sk_Y::V
    buffer_inner_Yk_X::V
    buffer_inner_Yk_Y::V
    last_gcd_result::Symbol
    last_gcd_stepsize::F
end

function get_parameter(d::QuasiNewtonLimitedMemoryBoxDirectionUpdate, ::Val{:max_stepsize})
    if d.last_gcd_result === :found_limited
        return d.last_gcd_stepsize
    else
        return Inf
    end
end

function QuasiNewtonLimitedMemoryBoxDirectionUpdate(
        qn_du::QuasiNewtonLimitedMemoryDirectionUpdate{<:AbstractQuasiNewtonUpdateRule, T, F}
    ) where {T, F <: Real}
    memory_size = capacity(qn_du.memory_s)
    M_11 = zeros(F, memory_size, memory_size)
    M_21 = zeros(F, memory_size, memory_size)
    M_22 = zeros(F, memory_size, memory_size)
    buffer_inner_Sk_X = zeros(F, memory_size)
    buffer_inner_Sk_Y = zeros(F, memory_size)
    buffer_inner_Yk_X = zeros(F, memory_size)
    buffer_inner_Yk_Y = zeros(F, memory_size)
    return QuasiNewtonLimitedMemoryBoxDirectionUpdate{
        typeof(qn_du), F, typeof(M_11), typeof(buffer_inner_Sk_X),
    }(
        qn_du,
        qn_du.initial_scale,
        M_11,
        M_21,
        M_22,
        buffer_inner_Sk_X,
        buffer_inner_Sk_Y,
        buffer_inner_Yk_X,
        buffer_inner_Yk_Y,
        :not_searched,
        NaN,
    )
end

function initialize_update!(ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    initialize_update!(ha.qn_du)
    ha.last_gcd_result = :not_searched
    return ha
end

function (d::QuasiNewtonLimitedMemoryBoxDirectionUpdate)(
        mp::AbstractManoptProblem, st
    )
    r = zero_vector(get_manifold(mp), get_iterate(st))
    return d(r, mp, st)
end
function (d::QuasiNewtonLimitedMemoryBoxDirectionUpdate)(
        r, mp::AbstractManoptProblem, st
    )
    d.qn_du(r, mp, st)
    M = get_manifold(mp)
    p = get_iterate(st)
    X = get_gradient(st)
    gcd = GeneralizedCauchyDirectionSubsolver(M, p, d)
    d.last_gcd_result, d.last_gcd_stepsize = find_generalized_cauchy_direction!(M, gcd, r, p, r, X)
    return r
end

get_update_vector_transport(u::QuasiNewtonLimitedMemoryBoxDirectionUpdate) = get_update_vector_transport(u.qn_du)

function get_at_bound_index(M::ProductManifold, X, b::Tuple{Int, Any})
    return get_at_bound_index(M.manifolds[b[1]], submanifold_component(M, X, Val(1)), b[2])
end

@doc raw"""
    hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X)

Compute ``⟨X, B X⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`.
"""
function hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X)
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    normX_sqr = norm(M, p, X)^2

    if m == 0 || num_nonzero_rho == 0
        return gh.qn_du.initial_scale \ normX_sqr
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.buffer_inner_Yk_X[ii] = inner(M, p, gh.qn_du.memory_y[i], X)
        gh.buffer_inner_Sk_X[ii] = gh.current_scale * inner(M, p, gh.qn_du.memory_s[i], X)

        ii += 1
    end
    buffer_inner_Yk = view(gh.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Sk = view(gh.buffer_inner_Sk_X, 1:num_nonzero_rho)

    return hessian_value_from_inner_products(gh, normX_sqr, buffer_inner_Yk, buffer_inner_Sk, buffer_inner_Yk, buffer_inner_Sk)
end

@doc raw"""
    hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector)

Compute ``⟨X, B X⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`, and `X` is the
[`UnitVector`](@ref).
"""
function hessian_value_diag(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector)
    b = X.index
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    if m == 0 || num_nonzero_rho == 0
        return inv(gh.qn_du.initial_scale)
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.buffer_inner_Yk_X[ii] = get_at_bound_index(M, gh.qn_du.memory_y[i], b)
        gh.buffer_inner_Sk_X[ii] = gh.current_scale * get_at_bound_index(M, gh.qn_du.memory_s[i], b)

        ii += 1
    end
    buffer_inner_Yk = view(gh.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Sk = view(gh.buffer_inner_Sk_X, 1:num_nonzero_rho)

    return hessian_value_from_inner_products(gh, one(eltype(gh.qn_du.ρ)), buffer_inner_Yk, buffer_inner_Sk, buffer_inner_Yk, buffer_inner_Sk)
end

@doc raw"""
    hessian_value(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector, Y)

Compute ``⟨X, B Y⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`, where `X` is the
[`UnitVector`](@ref).
"""
function hessian_value(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X::UnitVector, Y)
    b = X.index

    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    Yb = get_at_bound_index(M, Y, b)
    if m == 0 || num_nonzero_rho == 0
        return gh.qn_du.initial_scale * Yb
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.buffer_inner_Yk_X[ii] = get_at_bound_index(M, gh.qn_du.memory_y[i], b)
        gh.buffer_inner_Sk_X[ii] = gh.current_scale * get_at_bound_index(M, gh.qn_du.memory_s[i], b)

        gh.buffer_inner_Yk_Y[ii] = inner(M, p, gh.qn_du.memory_y[i], Y)
        gh.buffer_inner_Sk_Y[ii] = gh.current_scale * inner(M, p, gh.qn_du.memory_s[i], Y)
        ii += 1
    end
    buffer_inner_Yk_X = view(gh.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Yk_Y = view(gh.buffer_inner_Yk_Y, 1:num_nonzero_rho)
    buffer_inner_Sk_X = view(gh.buffer_inner_Sk_X, 1:num_nonzero_rho)
    buffer_inner_Sk_Y = view(gh.buffer_inner_Sk_Y, 1:num_nonzero_rho)

    return hessian_value_from_inner_products(gh, Yb, buffer_inner_Yk_X, buffer_inner_Sk_X, buffer_inner_Yk_Y, buffer_inner_Sk_Y)
end

@doc raw"""
    set_M_current_scale!(M::AbstractManifold, p, gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate)

Refresh the scaling factor and blockwise Hessian approximation stored in `gh` using the
nonzero curvature pairs currently in memory.

- Identifies the most recent index with nonzero ``ρ_i`` to scale the initial Hessian guess
    by ``ρ_i‖y_i‖^2 / θ``.
- Builds ``L_k`` and ``S_k^\top S_k`` from the stored ``(s_i, y_i)`` pairs and updates the
    block matrices ``M_{11}``, ``M_{21}``, and ``M_{22}`` via the blockwise inverse formula.
- If all ``ρ_i`` vanish, resets `current_scale` to the inverse of `initial_scale` and
    clears the block matrices.

Returns the mutated `gh`.
"""
function set_M_current_scale!(M::AbstractManifold, p, gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    m = length(gh.qn_du.memory_s)
    last_safe_index = -1
    for i in eachindex(gh.qn_du.ρ)
        if abs(gh.qn_du.ρ[i]) > 0
            last_safe_index = i
        end
    end

    if (last_safe_index == -1)
        # All memory yield zero inner products
        gh.current_scale = inv(gh.qn_du.initial_scale)
        gh.M_11 = fill(0.0, 0, 0)
        gh.M_21 = fill(0.0, 0, 0)
        gh.M_22 = fill(0.0, 0, 0)
        return gh
    end

    invA = Diagonal([-ri for ri in gh.qn_du.ρ if !iszero(ri)])
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    Lk = LowerTriangular(zeros(num_nonzero_rho, num_nonzero_rho))

    # total scaling factor for the initial Hessian
    # written this way to avoid floating point overflow (when ynorm is finite but ynorm^2 is Inf)
    # see CUTEst EXPQUAD problem for an example
    ynorm = norm(M, p, gh.qn_du.memory_y[last_safe_index])
    gh.current_scale = ((gh.qn_du.ρ[last_safe_index] * ynorm) * ynorm) / gh.qn_du.initial_scale

    tsksk = Symmetric(zeros(num_nonzero_rho, num_nonzero_rho))
    ii = 1
    # fill Dk and Lk
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        jj = 1
        for j in 1:m
            iszero(gh.qn_du.ρ[j]) && continue
            if jj < ii
                Lk[ii, jj] = inner(M, p, gh.qn_du.memory_s[i], gh.qn_du.memory_y[j])
            end
            if ii <= jj
                tsksk.data[ii, jj] = inner(M, p, gh.qn_du.memory_s[i], gh.qn_du.memory_s[j])
            end
            jj += 1
        end
        ii += 1
    end
    tsksk.data .*= gh.current_scale

    # matrix inversion using the blockwise formula for speed
    # Schur complement of -Dk is the only non-diagonal matrix we actually need to inverse in this step
    W1 = Lk * invA
    W2 = W1 * Lk'
    gh.M_22 = inv(Symmetric(tsksk - W2))
    W3 = gh.M_22 * W1
    W4 = W1' * W3

    gh.M_11 = invA + W4
    gh.M_21 = -W3

    return gh
end

@doc raw"""
    hessian_value_from_inner_products(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, iss::Real, cy1, cs1, cy2, cs2)

Evaluate the quadratic form defined by the current blockwise Hessian approximation stored in
`gh`, given precomputed coordinate vectors.

Arguments:
- `iss`: inner product of original vectors.
- `cy1`, `cy2`: coordinates of ``y``-like vectors in the ``Y_k`` basis.
- `cs1`, `cs2`: coordinates of ``s``-like vectors in the scaled ``S_k`` basis.

The result is ``θ·iss - cy₁ᵀ M₁₁ cy₂ - 2·cs₁ᵀ M₂₁ cy₂ - cs₁ᵀ M₂₂ cs₂`` using the blocks
``M₁₁``, ``M₂₁``, ``M₂₂`` stored in `gh` and the current scale ``θ``. Returns the scalar value.
"""
function hessian_value_from_inner_products(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, iss::Real, cy1, cs1, cy2, cs2)
    result = gh.current_scale * iss
    if length(cy1) == 0
        return result
    end
    result -= dot(cy1, gh.M_11, cy2)
    result -= 2 * dot(cs1, gh.M_21, cy2)
    result -= dot(cs1, gh.M_22, cs2)

    return result
end


@doc raw"""
    update_hessian!(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, p)

Update Hessian approximation `gh` by moving it to point `p` and updating the stored `s` and
`y` vectors.
"""
function update_hessian!(
        gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate,
        mp::AbstractManoptProblem,
        st::AbstractManoptSolverState,
        p_old,
        k::Int,
    )
    (capacity(gh.qn_du.memory_s) == 0) && return gh
    update_hessian!(gh.qn_du, mp, st, p_old, k)
    set_M_current_scale!(get_manifold(mp), get_iterate(st), gh)
    return gh
end


"""
    abstract type AbstractSegmentHessianUpdater end

Abstract type for methods that calculate f' and f'' in the GCD calculation in subsequent
line segments in [`GeneralizedCauchyDirectionSubsolver`](@ref).
"""
abstract type AbstractSegmentHessianUpdater end

"""
    init_updater!(::AbstractManifold, hessian_segment_updater::AbstractSegmentHessianUpdater, p, d, ha::AbstractQuasiNewtonDirectionUpdate)

Method for initialization of `AbstractSegmentHessianUpdater` `hessian_segment_updater` just before the loop
that examines subsequent intervals for GCD.
"""
init_updater!(::AbstractManifold, hessian_segment_updater::AbstractSegmentHessianUpdater, p, d, ha::AbstractQuasiNewtonDirectionUpdate)

"""
    struct GenericSegmentHessianUpdater <: AbstractSegmentHessianUpdater end

Generic f' and f'' calculation that only relies on `hessian_value` but is relatively slow for
high-dimensional domains.
"""
struct GenericSegmentHessianUpdater{TX} <: AbstractSegmentHessianUpdater
    d_z::TX
    d_tmp::TX
end

function get_default_hessian_segment_updater(M::AbstractManifold, p, ::AbstractQuasiNewtonDirectionUpdate)
    return GenericSegmentHessianUpdater(zero_vector(M, p), zero_vector(M, p))
end

function init_updater!(M::AbstractManifold, hessian_segment_updater::GenericSegmentHessianUpdater, p, d, ha::AbstractQuasiNewtonDirectionUpdate)
    zero_vector!(M, hessian_segment_updater.d_z, p)
    copyto!(M, hessian_segment_updater.d_tmp, d)
    return hessian_segment_updater
end

@doc raw"""
    (upd::GenericSegmentHessianUpdater)(M::AbstractManifold, p, t::Real, dt::Real, b, db, ha::AbstractQuasiNewtonDirectionUpdate)

Calculate Hessian values ``⟨e_b, B d_z⟩`` and ``⟨e_b, B d_tmp⟩`` for the generalized Cauchy
point line search using the generic approach via `hessian_value` with [`UnitVector`](@ref).
``d_z`` start with 0 and is updated in-place by adding `dt * d` to it.
"""
function (upd::GenericSegmentHessianUpdater)(M::AbstractManifold, p, t::Real, dt::Real, b, db, ha)
    upd.d_z .+= dt .* upd.d_tmp
    hv_eb_dz = hessian_value(ha, M, p, UnitVector(b), upd.d_z)
    hv_eb_d = hessian_value(ha, M, p, UnitVector(b), upd.d_tmp)

    set_zero_at_index!(M, upd.d_tmp, b)

    return hv_eb_dz, hv_eb_d
end

"""
    struct LimitedMemorySegmentHessianUpdater{TV <: AbstractVector} <: AbstractSegmentHessianUpdater

Hessian value calculation for generalized Cauchy direction line segments that is optimized for
[`QuasiNewtonLimitedMemoryBoxDirectionUpdate`](@ref). It relies on a specific Hessian structure.
"""
struct LimitedMemorySegmentHessianUpdater{TV <: AbstractVector} <: AbstractSegmentHessianUpdater
    p_s::TV
    p_y::TV
    c_s::TV
    c_y::TV
end

function get_default_hessian_segment_updater(::AbstractManifold, p, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    return LimitedMemorySegmentHessianUpdater(similar(ha.qn_du.ρ), similar(ha.qn_du.ρ), similar(ha.qn_du.ρ), similar(ha.qn_du.ρ))
end

function init_updater!(M::AbstractManifold, hessian_segment_updater::LimitedMemorySegmentHessianUpdater, p, d, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    fill!(hessian_segment_updater.c_s, 0)
    fill!(hessian_segment_updater.c_y, 0)
    ii = 1
    for i in eachindex(ha.qn_du.ρ)
        if iszero(ha.qn_du.ρ[i])
            continue
        end

        hessian_segment_updater.p_s[ii] = ha.current_scale * inner(M, p, ha.qn_du.memory_s[i], d)
        hessian_segment_updater.p_y[ii] = inner(M, p, ha.qn_du.memory_y[i], d)
        ii += 1
    end
    return hessian_segment_updater
end

@doc raw"""
    (hessian_segment_updater::LimitedMemorySegmentHessianUpdater)(
        M::AbstractManifold, p,
        t::Real, dt::Real, b, db, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate
    )

Calculate Hessian values ``⟨e_b, B d_z⟩`` and ``⟨e_b, B d⟩`` for the generalized Cauchy
point line search using the limited-memory block Hessian stored in `ha`.
``d_z`` start with 0 and is updated in-place by adding `dt * d` to it.

## Arguments:

- `M`: manifold.
- `p`: current iterate.
- `t`: current step length from `p`.
- `dt`: step length increment from the last step.
- `b`: bound index of the current segment.
- `db`: search direction component at the bound index `b`.

The updater reuses cached coordinate projections in `hessian_segment_updater` to cheaply
evaluate Hessian quadratic forms via `hessian_value_from_inner_products`.
"""
function (hessian_segment_updater::LimitedMemorySegmentHessianUpdater)(
        M::AbstractManifold, p,
        t::Real, dt::Real, b, db, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate
    )

    m = length(ha.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, ha.qn_du.ρ)

    ii = 1
    for i in 1:m
        iszero(ha.qn_du.ρ[i]) && continue
        # setting _X to w_b from the paper
        ha.buffer_inner_Yk_X[ii] = get_at_bound_index(M, ha.qn_du.memory_y[i], b)
        ha.buffer_inner_Sk_X[ii] = ha.current_scale * get_at_bound_index(M, ha.qn_du.memory_s[i], b)

        ii += 1
    end

    buffer_inner_Yk_eb = view(ha.buffer_inner_Yk_X, 1:num_nonzero_rho)
    buffer_inner_Sk_eb = view(ha.buffer_inner_Sk_X, 1:num_nonzero_rho)

    buffer_inner_cy = view(hessian_segment_updater.c_y, 1:num_nonzero_rho)
    buffer_inner_cs = view(hessian_segment_updater.c_s, 1:num_nonzero_rho)
    buffer_inner_py = view(hessian_segment_updater.p_y, 1:num_nonzero_rho)
    buffer_inner_ps = view(hessian_segment_updater.p_s, 1:num_nonzero_rho)

    buffer_inner_cy .+= dt .* buffer_inner_py
    buffer_inner_cs .+= dt .* buffer_inner_ps

    eb_B_z = hessian_value_from_inner_products(ha, t * db, buffer_inner_Yk_eb, buffer_inner_Sk_eb, buffer_inner_cy, buffer_inner_cs)

    eb_B_d = hessian_value_from_inner_products(ha, db, buffer_inner_Yk_eb, buffer_inner_Sk_eb, buffer_inner_py, buffer_inner_ps)

    buffer_inner_py .-= db .* buffer_inner_Yk_eb
    buffer_inner_ps .-= db .* buffer_inner_Sk_eb

    return eb_B_z, eb_B_d
end

struct ProductIndex{T <: Tuple}
    ranges::T
end

Base.iterate(itr::ProductIndex) = _iterate(itr.ranges, 1, nothing)
Base.iterate(itr::ProductIndex, state) = _iterate(itr.ranges, state...)

function _iterate(ranges, i, st)
    i > length(ranges) && return nothing
    if st === nothing
        it = iterate(ranges[i])
        it === nothing && return _iterate(ranges, i + 1, nothing)
        (j, st2) = it
        return ((i, j), (i, st2))
    else
        it = iterate(ranges[i], st)
        if it === nothing
            return _iterate(ranges, i + 1, nothing)
        else
            (j, st2) = it
            return ((i, j), (i, st2))
        end
    end
end


"""
    to_coordinate_index(M::ProductManifold, b::UnitVector, B::AbstractBasis)

Get the index of coordinate equal to 1 of [`UnitVector`](@ref) `b` with respect to
`AbstractBasis` `B`.
"""
to_coordinate_index(::AbstractManifold, b::UnitVector{Int}, ::AbstractBasis) = b.index
"""
    to_coordinate_index(M::ProductManifold, b::UnitVector, B::AbstractBasis)

Get the index of coordinate equal to 1 of [`UnitVector`](@ref) `b` with respect to
`AbstractBasis` `B`.
"""
function to_coordinate_index(M::ProductManifold, b::UnitVector{Tuple{Int, Int}}, B::AbstractBasis)
    i, j = b.index
    offset = sum(k -> number_of_coordinates(M.manifolds[k], B), 1:(i - 1); init = 0)
    return offset + j
end

Base.length(itr::ProductIndex) = sum(length, itr.ranges)


"""
    get_bounds_index(::AbstractManifold)

Get the bound indices of manifold `M`. Standard manifolds don't have bounds, so
`Base.OneTo(1)` is returned.
"""
get_bounds_index(M::AbstractManifold) = Base.OneTo(0)
function get_bounds_index(M::ProductManifold)
    ranges = map(get_bounds_index, M.manifolds)
    iter = ProductIndex(ranges)
    return iter
end

"""
    get_stepsize_bound(M::AbstractManifold, x, d, i)

Get the upper bound on moving in direction `d` from point `p` on manifold `M`, for the
bound index `i`.
"""
get_stepsize_bound(M::AbstractManifold, p, d, i)
function get_stepsize_bound(M::ProductManifold, p, d, i::Tuple{Int, Any})
    i1, i2 = i
    return get_stepsize_bound(M.manifolds[i1], submanifold_component(M, p, i1), submanifold_component(M, d, i1), i2)
end

"""
    set_zero_at_index!(M::ProductManifold, d, i::Tuple{Int,Any})

Set the element of the `i[1]`th component of `d` at bound index `i[2]` to zero.
"""
function set_zero_at_index!(M::ProductManifold, d, i::Tuple{Int, Any})
    i1, i2 = i
    set_zero_at_index!(M.manifolds[i1], submanifold_component(M, d, i1), i2)
    return d
end

"""
    set_stepsize_bound!(M::AbstractManifold, d_out, p, d, t_current::Real)

For each component at index `i` in the tangent vector `d_out`, if the stepsize bound in
direction `d` for that component is less than `t_current`, set the element of `d_out` to
the distance from `p[i]` to the bound in the direction of `d[i]`. If the stepsize bound is
non-positive, set the element to 0.

By default it does not modify `d_out` because most manifolds don't have direction-specific
stepsize bounds, and general anisotropic bounds are handled differently.
"""
function set_stepsize_bound!(::AbstractManifold, d_out, p, d, t_current::Real)
    return d_out
end

function set_stepsize_bound!(M::ProductManifold, d_out, p, d, t_current::Real)
    map(
        (N, d_out_c, p_c, d_c) -> set_stepsize_bound!(N, d_out_c, p_c, d_c, t_current),
        M.manifolds,
        submanifold_components(M, d_out),
        submanifold_components(M, p),
        submanifold_components(M, d),
    )
    return d_out
end

@doc raw"""
    GeneralizedCauchyDirectionSubsolver{TM <: AbstractManifold, TP, T_HA <: AbstractQuasiNewtonDirectionUpdate, TFU <: AbstractSegmentHessianUpdater}

Helper container for generalized Cauchy direction search. Stores the manifold `M`, cached
original descent direction (`d_original`), the quasi-Newton direction update `ha`, and the
`hessian_segment_updater`, which computes certain values of the Hessian while advancing segments.
Instances are reused across segments during [`find_generalized_cauchy_direction!`](@ref) to
avoid allocations.
"""
struct GeneralizedCauchyDirectionSubsolver{
        TX,
        T_HA <: AbstractQuasiNewtonDirectionUpdate, TFU <: AbstractSegmentHessianUpdater, TFT <: Tuple{<:Real, Any}, TBI,
        TO <: Base.Order.Ordering,
    }
    d_original::TX
    ha::T_HA
    hessian_segment_updater::TFU
    F_list::Vector{TFT}
    bounds_indices::TBI
    ordering::TO
end

function GeneralizedCauchyDirectionSubsolver(
        M::AbstractManifold, p, ha::AbstractQuasiNewtonDirectionUpdate;
        hessian_segment_updater::AbstractSegmentHessianUpdater = get_default_hessian_segment_updater(M, p, ha)
    )
    bounds_indices = get_bounds_index(M)
    TInd = eltype(bounds_indices)
    TF = number_eltype(p)
    F_list = Tuple{TF, TInd}[]
    sizehint!(F_list, length(bounds_indices) + 1)
    ordering = Base.By(first)
    return GeneralizedCauchyDirectionSubsolver(
        zero_vector(M, p), ha,
        hessian_segment_updater, F_list, bounds_indices, ordering
    )
end

function collect_isotropic_limits!(::AbstractManifold, ::Vector{<:Tuple{TF, Any}}, p, d)::Tuple{Bool, TF} where {TF <: Real}
    return false, convert(TF, Inf)
end

function collect_isotropic_limits!(M::ProductManifold, F_list::Vector{<:Tuple{TF, Any}}, p, d)::Tuple{Bool, TF} where {TF <: Real}
    has_finite_limit = false
    smallest_positive_limit = Inf
    map(M.manifolds, submanifold_components(M, p), submanifold_components(M, d)) do Mi, p_i, d_i
        if !has_anisotropic_max_stepsize(Mi)
            max_step = Manopt.max_stepsize(Mi, p_i)
            if isfinite(max_step)
                tms = max_step / norm(Mi, p_i, d_i)
                push!(F_list, (tms, -1))
                has_finite_limit = true
                if tms < smallest_positive_limit
                    smallest_positive_limit = tms
                end
            end
        end
    end
    return has_finite_limit, smallest_positive_limit
end

"""
    find_generalized_cauchy_direction!(
        M::AbstractManifold,
        gcd::GeneralizedCauchyDirectionSubsolver, d_out, p, d, X
    )

Find generalized Cauchy direction looking from point `p` on manifold `M` in direction `d`
and save it to `d_out`. Gradient of the objective at `p` is `X`.

The function returns a pair (status, max_stepsize) where `status` is a symbol describing
the result of the search, and `max_stepsize` is the maximum stepsize that can be taken in
the direction `d_out`.

The `status` can be one of the following:
* `:found_limited` if the point was found and we can perform a step of length at most 1
  in direction `d_out` afterwards,
* `:found_unlimited` if the point was found and we can perform a step of length at most
  `max_stepsize(M, p)` in direction `d_out` afterwards,
* `:not_found` if the search cannot be performed in direction `d`.
"""
function find_generalized_cauchy_direction!(
        M::AbstractManifold,
        gcd::GeneralizedCauchyDirectionSubsolver{
            <:Any, <:AbstractQuasiNewtonDirectionUpdate,
            <:AbstractSegmentHessianUpdater, <:Tuple{TF, Any},
        },
        d_out, p, d, X
    ) where {TF <: Real}
    copyto!(M, gcd.d_original, d)
    copyto!(M, d_out, d)

    ordering = gcd.ordering
    F_list = gcd.F_list
    empty!(F_list)

    bounds_indices = gcd.bounds_indices

    # isotropic limits
    has_finite_limit, smallest_positive_limit = collect_isotropic_limits!(M, F_list, p, d)
    # anisotropic limits
    for i in bounds_indices
        sbi = get_stepsize_bound(M, p, d, i)::TF

        if sbi > 0
            push!(F_list, (sbi, i))
            if sbi < smallest_positive_limit
                smallest_positive_limit = sbi
            end
        end
        has_finite_limit |= isfinite(sbi)
    end

    # In this case we can't move in the direction `d` at all, though it's usually not
    # a problem relevant to the end user because it can be handled by step_solver! that
    # uses the GCD subsolver.
    if isempty(F_list)
        return (:not_found, NaN)
    end
    heapify!(F_list, ordering)

    f_prime = inner(M, p, X, d)
    f_double_prime = hessian_value_diag(gcd.ha, M, p, d)

    if iszero(f_prime) || iszero(f_double_prime)
        return (:not_found, NaN)
    end

    dt_min = -f_prime / f_double_prime
    t_old = 0.0

    t_current, b = heappop!(F_list, ordering)
    dt = t_current - t_old

    init_updater!(M, gcd.hessian_segment_updater, p, d, gcd.ha)
    # b can be -1 if it corresponds to the max stepsize limit on the manifold part
    while dt_min > dt && b != -1
        db = get_at_bound_index(M, d, b)::TF
        gb = get_at_bound_index(M, X, b)::TF

        hv_eb_dz, hv_eb_d = gcd.hessian_segment_updater(M, p, t_current, dt, b, db, gcd.ha)::Tuple{TF, TF}

        f_prime += dt * f_double_prime - db * (gb + hv_eb_dz)
        f_double_prime += (2 * -db * hv_eb_d) + db^2 * hessian_value_diag(gcd.ha, M, p, UnitVector(b))

        t_old = t_current

        # If f_prime is 0, we've found the local minimizer (GCD)
        if iszero(f_prime) || iszero(f_double_prime)
            # It means that GCD is at the beginning of the t_current, so we want to set dt_min to 0 (stay in the point)
            dt_min = 0.0
            break
        end

        dt_min = -f_prime / f_double_prime
        isempty(F_list) && break

        t_current, b = heappop!(F_list, ordering)
        dt = t_current - t_old
    end

    dt_min = max(dt_min, 0.0)
    t_old = t_old + dt_min
    d_out .*= t_old
    # by construction, there is no bound achievable before stepsize 1.0 in direction d_out
    # there first bound after that is achieved at smallest_positive_limit / t_old
    max_feasible_stepsize = max(1.0, smallest_positive_limit / t_old)

    set_stepsize_bound!(M, d_out, p, gcd.d_original, t_old)
    if has_finite_limit
        return (:found_limited, max_feasible_stepsize)
    else
        return (:found_unlimited, Inf)
    end
end

"""
    struct MaxStepsizeInDirectionSubsolver end

Helper container for finding the maximum stepsize in a direction. Stores the manifold `M`,
container for the list of bounds `F_list`, and the bound indices.

## Constructor

    MaxStepsizeInDirectionSubsolver(M::AbstractManifold, p)

Initialize the `MaxStepsizeInDirectionSubsolver` for manifold `M` and point `p`. The `F_list`
is initialized to be empty and will be populated during the search for the maximum stepsize
in a direction. Floating point type of the elements bounds in `F_list` is determined by the
number type of `p`.

The `MaxStepsizeInDirectionSubsolver` can be reused for multiple different points and
directions on the same manifold, but it is not thread-safe.
"""
struct MaxStepsizeInDirectionSubsolver{TFT <: Tuple{<:Real, Any}, TBI}
    F_list::Vector{TFT}
    bounds_indices::TBI
end
function MaxStepsizeInDirectionSubsolver(M::AbstractManifold, p)
    bounds_indices = get_bounds_index(M)
    TInd = eltype(bounds_indices)
    TF = number_eltype(p)
    F_list = Tuple{TF, TInd}[]
    sizehint!(F_list, length(bounds_indices) + 1)
    return MaxStepsizeInDirectionSubsolver{Tuple{TF, TInd}, typeof(bounds_indices)}(F_list, bounds_indices)
end

"""
    find_max_stepsize_in_direction(M::AbstractManifold, gcd::MaxStepsizeInDirectionSubsolver, p, d)

Find the maximum stepsize that can be performed from point `p` in direction `d`.

The function returns a pair (status, max_stepsize) where `status` is a symbol describing
the result of the search, and `max_stepsize` is the maximum stepsize that can be taken in
the direction `d_out`.

The `status` can be one of the following:
* `:found_limited` if the point was found and we can perform a step of length at most 1
  in direction `d_out` afterwards,
* `:found_unlimited` if the point was found and we can perform a step of length at most
  `max_stepsize(M, p)` in direction `d_out` afterwards,
* `:not_found` if the search cannot be performed in direction `d`.
"""
function find_max_stepsize_in_direction(
        M::AbstractManifold,
        sdf::MaxStepsizeInDirectionSubsolver{<:Tuple{TF, Any}},
        p, d
    ) where {TF <: Real}

    F_list = sdf.F_list
    empty!(F_list)
    bounds_indices = sdf.bounds_indices

    # isotropic limits
    has_finite_limit, smallest_positive_limit = collect_isotropic_limits!(M, F_list, p, d)
    # anisotropic limits
    for i in bounds_indices
        sbi = get_stepsize_bound(M, p, d, i)::TF

        if sbi > 0
            push!(F_list, (sbi, i))
            if sbi < smallest_positive_limit
                smallest_positive_limit = sbi
            end
        end
        has_finite_limit |= isfinite(sbi)
    end

    if isempty(F_list)
        return (:not_found, NaN)
    end
    if has_finite_limit
        return (:found_limited, smallest_positive_limit)
    else
        return (:found_unlimited, Inf)
    end

end

function show(io::IO, qns::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    print(io, "QuasiNewtonLimitedMemoryBoxDirectionUpdate with internal state:\n")
    return print(io, qns.qn_du)
end
