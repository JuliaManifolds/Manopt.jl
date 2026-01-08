"""
    requires_generalized_cauchy_point_computation(M::AbstractManifold)

Return `true` if `M` is a `Hyperrectangle`-like manifold with corners, or a product of it
with a standard manifold. Otherwise return `false`.
"""
requires_generalized_cauchy_point_computation(::AbstractManifold) = false
requires_generalized_cauchy_point_computation(M::ProductManifold) = requires_generalized_cauchy_point_computation(M.manifolds[1])

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
    coords_Sk_X::V
    coords_Sk_Y::V
    coords_Yk_X::V
    coords_Yk_Y::V
    last_gcp_result::Symbol
end

function get_parameter(d::QuasiNewtonLimitedMemoryBoxDirectionUpdate, ::Val{:max_stepsize})
    if d.last_gcp_result === :found_limited
        return 1.0
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
    coords_Sk_X = zeros(F, memory_size)
    coords_Sk_Y = zeros(F, memory_size)
    coords_Yk_X = zeros(F, memory_size)
    coords_Yk_Y = zeros(F, memory_size)
    return QuasiNewtonLimitedMemoryBoxDirectionUpdate{
        typeof(qn_du), F, typeof(M_11), typeof(coords_Sk_X),
    }(
        qn_du,
        qn_du.initial_scale,
        M_11,
        M_21,
        M_22,
        coords_Sk_X,
        coords_Sk_Y,
        coords_Yk_X,
        coords_Yk_Y,
        :not_searched,
    )
end

function initialize_update!(ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    initialize_update!(ha.qn_du)
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
    gcp = GeneralizedCauchyPointFinder(M, p, d)
    d.last_gcp_result = find_generalized_cauchy_point_direction!(gcp, r, p, r, X)
    return r
end

get_update_vector_transport(u::QuasiNewtonLimitedMemoryBoxDirectionUpdate) = get_update_vector_transport(u.qn_du)

function get_at_bound_index(M::ProductManifold, X, b)
    return get_at_bound_index(M.manifolds[1], submanifold_component(M, X, Val(1)), b)
end

@doc raw"""
    hess_val(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X)

Compute ``⟨X, B X⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`.
"""
function hess_val(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, X)
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    normX_sqr = norm(M, p, X)^2

    if m == 0 || num_nonzero_rho == 0
        return gh.qn_du.initial_scale \ normX_sqr
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.coords_Yk_X[ii] = inner(M, p, gh.qn_du.memory_y[i], X)
        gh.coords_Sk_X[ii] = gh.current_scale * inner(M, p, gh.qn_du.memory_s[i], X)

        ii += 1
    end
    coords_Yk = view(gh.coords_Yk_X, 1:num_nonzero_rho)
    coords_Sk = view(gh.coords_Sk_X, 1:num_nonzero_rho)

    return hess_val_from_wmwt_coords(gh, normX_sqr, coords_Yk, coords_Sk, coords_Yk, coords_Sk)
end

@doc raw"""
    hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, b)

Compute ``⟨X, B X⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`, and `X` is the
unit vector along index `b`.
"""
function hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, b)
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    if m == 0 || num_nonzero_rho == 0
        return inv(gh.qn_du.initial_scale)
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.coords_Yk_X[ii] = get_at_bound_index(M, gh.qn_du.memory_y[i], b)
        gh.coords_Sk_X[ii] = gh.current_scale * get_at_bound_index(M, gh.qn_du.memory_s[i], b)

        ii += 1
    end
    coords_Yk = view(gh.coords_Yk_X, 1:num_nonzero_rho)
    coords_Sk = view(gh.coords_Sk_X, 1:num_nonzero_rho)

    return hess_val_from_wmwt_coords(gh, one(eltype(gh.qn_du.ρ)), coords_Yk, coords_Sk, coords_Yk, coords_Sk)
end

@doc raw"""
    hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, b, Y)

Compute ``⟨X, B Y⟩``, where ``B`` is the (1, 1)-Hessian represented by `gh`, where `X` is the
unit vector pointing at index `b`.
"""
function hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, M::AbstractManifold, p, b, Y)
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    Yb = get_at_bound_index(M, Y, b)
    if m == 0 || num_nonzero_rho == 0
        return gh.qn_du.initial_scale * Yb
    end

    ii = 1
    for i in 1:m
        iszero(gh.qn_du.ρ[i]) && continue
        gh.coords_Yk_X[ii] = get_at_bound_index(M, gh.qn_du.memory_y[i], b)
        gh.coords_Sk_X[ii] = gh.current_scale * get_at_bound_index(M, gh.qn_du.memory_s[i], b)

        gh.coords_Yk_Y[ii] = inner(M, p, gh.qn_du.memory_y[i], Y)
        gh.coords_Sk_Y[ii] = gh.current_scale * inner(M, p, gh.qn_du.memory_s[i], Y)
        ii += 1
    end
    coords_Yk_X = view(gh.coords_Yk_X, 1:num_nonzero_rho)
    coords_Yk_Y = view(gh.coords_Yk_Y, 1:num_nonzero_rho)
    coords_Sk_X = view(gh.coords_Sk_X, 1:num_nonzero_rho)
    coords_Sk_Y = view(gh.coords_Sk_Y, 1:num_nonzero_rho)

    return hess_val_from_wmwt_coords(gh, Yb, coords_Yk_X, coords_Sk_X, coords_Yk_Y, coords_Sk_Y)
end

@doc raw"""
    set_M_current_scale!(M::AbstractManifold, p, gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate)

Refresh the scaling factor and blockwise Hessian approximation stored in `gh` using the
nonzero curvature pairs currently in memory.

- Identifies the most recent index with nonzero ``ρ_i`` to scale the initial Hessian guess
    by ``ρ_i‖y_i‖^2 / θ``.
- Builds ``L_k`` and ``S_k^\top S_k`` from the stored ``(s_i, y_i)`` pairs and updates the
    block matrices ``M_{11}``, ``M_{21}``, and ``M_{22}`` via the blockwise inverse formula.
- If all ``ρ_i`` vanish, resets `current_scale` to `initial_scale` and clears the block
    matrices.

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
        gh.current_scale = gh.qn_du.initial_scale
        gh.M_11 = fill(0.0, 0, 0)
        gh.M_21 = fill(0.0, 0, 0)
        gh.M_22 = fill(0.0, 0, 0)
        return gh
    end

    invA = Diagonal([-ri for ri in gh.qn_du.ρ if !iszero(ri)])
    num_nonzero_rho = count(!iszero, gh.qn_du.ρ)

    Lk = LowerTriangular(zeros(num_nonzero_rho, num_nonzero_rho))

    # total scaling factor for the initial Hessian
    gh.current_scale = (gh.qn_du.ρ[last_safe_index] * norm(M, p, gh.qn_du.memory_y[last_safe_index])^2) / gh.qn_du.initial_scale

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
    hess_val_from_wmwt_coords(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, iss::Real, cy1, cs1, cy2, cs2)

Evaluate the quadratic form defined by the current blockwise Hessian approximation stored in
`gh`, given precomputed coordinate vectors.

Arguments:
- `iss`: inner product of original vectors.
- `cy1`, `cy2`: coordinates of ``y``-like vectors in the ``Y_k`` basis.
- `cs1`, `cs2`: coordinates of ``s``-like vectors in the scaled ``S_k`` basis.

The result is ``θ·iss - cy₁ᵀ M₁₁ cy₂ - 2·cs₁ᵀ M₂₁ cy₂ - cs₁ᵀ M₂₂ cs₂`` using the blocks
``M₁₁``, ``M₂₁``, ``M₂₂`` stored in `gh` and the current scale ``θ``. Returns the scalar value.
"""
function hess_val_from_wmwt_coords(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, iss::Real, cy1, cs1, cy2, cs2)
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
    abstract type AbstractFPFPPUpdater end

Abstract type for methods that calculate f' and f'' in the GCP calculation in subsequent
line segments in `GeneralizedCauchyPointFinder`.
"""
abstract type AbstractFPFPPUpdater end

"""
    init_updater!(::AbstractManifold, fpfpp_upd::AbstractFPFPPUpdater, p, d, ha::AbstractQuasiNewtonDirectionUpdate)

Method for initialization of `AbstractFPFPPUpdater` `fpfpp_upd` just before the loop
that examines subsequent intervals for GCP. By default it does nothing.
"""
init_updater!(::AbstractManifold, fpfpp_upd::AbstractFPFPPUpdater, p, d, ha::AbstractQuasiNewtonDirectionUpdate) = fpfpp_upd

"""
    struct GenericFPFPPUpdater <: AbstractFPFPPUpdater end

Generic f' and f'' calculation that only relies on `hess_val_eb` but is relatively slow for
high-dimensional domains.
"""
struct GenericFPFPPUpdater <: AbstractFPFPPUpdater end

get_default_fpfpp_updater(::AbstractQuasiNewtonDirectionUpdate) = GenericFPFPPUpdater()

"""
    struct LimitedMemoryFPFPPUpdater{TV <: AbstractVector} <: AbstractFPFPPUpdater

f' and f'' calculation that is optimized for `QuasiNewtonLimitedMemoryBoxDirectionUpdate`.
It relies on a specific Hessian structure.
"""
struct LimitedMemoryFPFPPUpdater{TV <: AbstractVector} <: AbstractFPFPPUpdater
    p_s::TV
    p_y::TV
    c_s::TV
    c_y::TV
end

function get_default_fpfpp_updater(ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    return LimitedMemoryFPFPPUpdater(similar(ha.qn_du.ρ), similar(ha.qn_du.ρ), similar(ha.qn_du.ρ), similar(ha.qn_du.ρ))
end

function (::GenericFPFPPUpdater)(M::AbstractManifold, p, old_f_prime, old_f_double_prime, dt, db, gb, ha, b, z, d_old)
    f_prime = old_f_prime + dt * old_f_double_prime - db * (gb + hess_val_eb(ha, M, p, b, z))
    f_double_prime = old_f_double_prime + (2 * -db * hess_val_eb(ha, M, p, b, d_old)) + db^2 * hess_val_eb(ha, M, p, b)

    return f_prime, f_double_prime
end

function init_updater!(M::AbstractManifold, fpfpp_upd::LimitedMemoryFPFPPUpdater, p, d, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    fill!(fpfpp_upd.c_s, 0)
    fill!(fpfpp_upd.c_y, 0)
    ii = 1
    for i in eachindex(ha.qn_du.ρ)
        if iszero(ha.qn_du.ρ[i])
            continue
        end

        fpfpp_upd.p_s[ii] = ha.current_scale * inner(M, p, ha.qn_du.memory_s[i], d)
        fpfpp_upd.p_y[ii] = inner(M, p, ha.qn_du.memory_y[i], d)
        ii += 1
    end
    return fpfpp_upd
end

@doc raw"""
    (fpfpp_upd::LimitedMemoryFPFPPUpdater)(
        M::AbstractManifold, p, old_f_prime::Real, old_f_double_prime::Real,
        dt::Real, db, gb, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b, z, d_old
    )

Update ``f'`` and ``f''`` for the generalized Cauchy point line search using the limited-memory
block Hessian stored in `ha`.

## Arguments:

- `old_f_prime`, `old_f_double_prime`: values carried from the previous segment.
- `dt`: step length along the segment direction.
- `db`: direction component at the bound index `b`.
- `gb`: gradient component at the bound index `b`.
- `z`: trial step vector.
- `d_old`: previous search direction.

The updater reuses cached coordinate projections in `fpfpp_upd` to cheaply evaluate Hessian
quadratic forms via `hess_val_from_wmwt_coords`, then returns the new `(f', f'')` pair.
"""
function (fpfpp_upd::LimitedMemoryFPFPPUpdater)(
        M::AbstractManifold, p, old_f_prime::Real, old_f_double_prime::Real,
        dt::Real, db, gb, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b, z, d_old
    )

    m = length(ha.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, ha.qn_du.ρ)

    iss_eb_z = get_at_bound_index(M, z, b)
    iss_eb_d = get_at_bound_index(M, d_old, b)

    ii = 1
    for i in 1:m
        iszero(ha.qn_du.ρ[i]) && continue
        # setting _X to w_b from the paper
        ha.coords_Yk_X[ii] = get_at_bound_index(M, ha.qn_du.memory_y[i], b)
        ha.coords_Sk_X[ii] = ha.current_scale * get_at_bound_index(M, ha.qn_du.memory_s[i], b)

        ii += 1
    end

    coords_Yk_eb = view(ha.coords_Yk_X, 1:num_nonzero_rho)
    coords_Sk_eb = view(ha.coords_Sk_X, 1:num_nonzero_rho)

    coords_cy = view(fpfpp_upd.c_y, 1:num_nonzero_rho)
    coords_cs = view(fpfpp_upd.c_s, 1:num_nonzero_rho)
    coords_py = view(fpfpp_upd.p_y, 1:num_nonzero_rho)
    coords_ps = view(fpfpp_upd.p_s, 1:num_nonzero_rho)

    coords_cy .+= dt .* coords_py
    coords_cs .+= dt .* coords_ps

    eb_B_z = hess_val_from_wmwt_coords(ha, iss_eb_z, coords_Yk_eb, coords_Sk_eb, coords_cy, coords_cs)

    f_prime = old_f_prime + dt * old_f_double_prime - db * (gb + eb_B_z)
    eb_B_d = hess_val_from_wmwt_coords(ha, iss_eb_d, coords_Yk_eb, coords_Sk_eb, coords_py, coords_ps)

    f_double_prime = old_f_double_prime - 2 * db * eb_B_d + db^2 * hess_val_eb(ha, M, p, b)

    coords_py .-= db .* coords_Yk_eb
    coords_ps .-= db .* coords_Sk_eb

    return f_prime, f_double_prime
end

"""
    get_bounds_index(::AbstractManifold)

Get the bound indices of manifold `M`. Standard manifolds don't have bounds, so
`Base.OneTo(1)` is returned.
"""
get_bounds_index(M::AbstractManifold)
get_bounds_index(M::ProductManifold) = get_bounds_index(M.manifolds[1])

"""
    get_bound_t(M::AbstractManifold, x, d, i)

Get the upper bound on moving in direction `d` from point `p` on manifold `M`, for the
bound index `i`.
"""
get_bound_t(M::AbstractManifold, p, d, i)
function get_bound_t(M::ProductManifold, p, d, i)
    return get_bound_t(M.manifolds[1], submanifold_component(M, p, Val(1)), submanifold_component(M, d, Val(1)), i)
end
function set_bound_t_at_index!(M::ProductManifold, p_cp, t, d, i)
    set_bound_t_at_index!(M.manifolds[1], submanifold_component(M, p_cp, Val(1)), t, d, i)
    return p_cp
end

function set_bound_at_index!(M::ProductManifold, p_cp, d, i)
    set_bound_at_index!(M.manifolds[1], submanifold_component(M, p_cp, Val(1)), submanifold_component(M, d, Val(1)), i)
    return p_cp
end

@doc raw"""
    bound_direction_tweak!(M::ProductManifold, d_out, d, p, p_cp)

Set `d_out .= p_cp .- p` on the `Hyperrectangle` part of the `ProductManifold` `M`.

Return the mutated `d_out`.
"""
function bound_direction_tweak!(M::ProductManifold, d_out, d, p, p_cp)
    bound_direction_tweak!(
        M.manifolds[1], submanifold_component(M, d_out, Val(1)),
        submanifold_component(M, d, Val(1)), submanifold_component(M, p, Val(1)),
        submanifold_component(M, p_cp, Val(1))
    )
    return d_out
end


@doc raw"""
    GeneralizedCauchyPointFinder{TM <: AbstractManifold, TP, TX, T_HA <: AbstractQuasiNewtonDirectionUpdate, TFU <: AbstractFPFPPUpdater}

Helper container for generalized Cauchy point search. Stores the manifold `M`, cached
workspace (`p_cp`, `Y_tmp`, `d_old`), the quasi-Newton direction update `ha`, and the
``f'``/``f''`` updater `fpfpp_updater`. Instances are reused across segments during
`find_generalized_cauchy_point_direction!` to avoid allocations.
"""
struct GeneralizedCauchyPointFinder{TM <: AbstractManifold, TP, TX, T_HA <: AbstractQuasiNewtonDirectionUpdate, TFU <: AbstractFPFPPUpdater}
    M::TM
    p_cp::TP
    Y_tmp::TX
    d_old::TX
    ha::T_HA
    fpfpp_updater::TFU
end

function GeneralizedCauchyPointFinder(
        M::AbstractManifold, p, ha::AbstractQuasiNewtonDirectionUpdate;
        fpfpp_updater::AbstractFPFPPUpdater = get_default_fpfpp_updater(ha)
    )
    return GeneralizedCauchyPointFinder(M, copy(M, p), zero_vector(M, p), zero_vector(M, p), ha, fpfpp_updater)
end

"""
    find_generalized_cauchy_point_direction!(gcp::GeneralizedCauchyPointFinder, d_out, p, d, X)

Find generalized Cauchy point looking from point `p` in direction `d` and save the tangent
vector pointing at it to `d_out`. Gradient of the objective at `p` is `X`.

The function returns 
* `:found_limited` if the point was found and we can perform a step of length at most 1
  in direction `d_out` afterwards,
* `:found_unlimited` if the point was found and we can perform a step of length at most
  `max_stepsize(M, p)` in direction `d_out` afterwards,
* `:not_found` if the search cannot be performed in direction `d`.
"""
function find_generalized_cauchy_point_direction!(gcp::GeneralizedCauchyPointFinder, d_out, p, d, X)
    M = gcp.M
    copyto!(M, gcp.p_cp, p)
    p_cp = gcp.p_cp
    zero_vector!(M, gcp.Y_tmp, p)
    copyto!(M, d_out, d)

    bounds_indices = get_bounds_index(M)
    TInd = eltype(bounds_indices)
    TF = number_eltype(d)

    t = Dict{TInd, TF}((k, Inf) for k in bounds_indices)

    F_list = Tuple{TF, TInd}[]
    sizehint!(F_list, length(bounds_indices) + 1)

    has_finite_limit = false

    for i in bounds_indices
        t[i] = get_bound_t(M, p, d, i)

        if t[i] > 0
            push!(F_list, (t[i], i))
        end
        has_finite_limit |= isfinite(t[i])
    end

    if M isa ProductManifold
        # Hyperrectangle × something else
        # push also `t` corresponding to max_stepsize if it is considered in the manifold
        M2 = M.manifolds[2]
        p2 = submanifold_component(M, p, Val(2))
        max_step = Manopt.max_stepsize(M2, p2)
        if isfinite(max_step)
            d2 = submanifold_component(M, d, Val(2))
            tms = max_step / norm(M2, p2, d2)
            push!(F_list, (tms, -1))
        end
    else
        # Check only when we work on a pure Hyperrectangle
        if isempty(F_list)
            @warn "We can't go in the selected direction"
            return :not_found
        end
    end

    F = BinaryHeap(Base.By(first), F_list)

    f_prime = inner(M, p, X, d)
    f_double_prime = hess_val(gcp.ha, M, p, d)

    if iszero(f_prime) || iszero(f_double_prime)
        return :not_found
    end

    dt_min = -f_prime / f_double_prime
    t_old = 0.0

    t_current, b = pop!(F)
    dt = t_current - t_old

    init_updater!(M, gcp.fpfpp_updater, p, d, gcp.ha)
    # b can be -1 if it corresponds to the max stepsize limit on the manifold part
    while dt_min > dt && b != -1
        gcp.Y_tmp .+= dt .* d
        copyto!(M, gcp.d_old, d)
        set_bound_at_index!(M, p_cp, d, b)

        gb = get_at_bound_index(M, X, b)
        db = get_at_bound_index(M, gcp.d_old, b)

        f_prime, f_double_prime = gcp.fpfpp_updater(M, p, f_prime, f_double_prime, dt, db, gb, gcp.ha, b, gcp.Y_tmp, gcp.d_old)
        t_old = t_current

        # If f_prime is 0, we've found the local minimizer (GCP)
        if iszero(f_prime) || iszero(f_double_prime)
            # It means that GCP is at the beginning of the t_current, so we want to set dt_min to 0 (stay in the point)
            dt_min = 0.0
            break
        end

        dt_min = -f_prime / f_double_prime
        isempty(F) && break

        t_current, b = pop!(F)
        dt = t_current - t_old
    end

    dt_min = max(dt_min, 0.0)
    t_old = t_old + dt_min
    for i in bounds_indices
        if t[i] >= t_current
            set_bound_t_at_index!(M, p_cp, t_old, d, i)
        end
    end

    bound_direction_tweak!(M, d_out, d, p, p_cp)

    if has_finite_limit
        return :found_limited
    else
        return :found_unlimited
    end
end
