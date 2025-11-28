mutable struct QuasiNewtonLimitedMemoryBoxDirectionUpdate{
        TDU <: QuasiNewtonLimitedMemoryDirectionUpdate,
        TM <: AbstractManifold,
        F <: Real,
        THM <: AbstractMatrix,
        V <: AbstractVector,
    } <: AbstractQuasiNewtonDirectionUpdate
    # this approximates inverse Hessian
    qn_du::TDU

    # fields for approximating the Hessian
    current_scale::F
    M_11::THM
    M_21::THM
    M_22::THM
    # buffer for calculating stuff
    coords_Sk_X::V
    coords_Sk_Y::V
    coords_Yk_X::V
    coords_Yk_Y::V
end

function initialize_update!(ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    initialize_update!(ha.qn_du)
    return ha
end

@doc raw"""
    hess_val(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, X)

Compute $⟨X, B X⟩$, where $B$ is the (1, 1)-Hessian represented by `gh`.
"""
hess_val(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, X) = hess_val_single_pass(gh, X)

@doc raw"""
    hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b)

Compute $⟨X, B X⟩$, where $B$ is the (1, 1)-Hessian represented by `gh`, and `X` is the
unit vector along index `b`.
"""
hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b) = hess_val_single_pass_eb(gh, b)

@doc raw"""
    hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b, Y)

Compute $⟨X, B Y⟩$, where $B$ is the (1, 1)-Hessian represented by `gh`, where `X` is the
unit vector pointing at index `b`.
"""
hess_val_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b, Y) = hess_val_single_pass_eb(gh, b, Y)

function set_M_current_scale!(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    M = gh.M
    p = gh.p
    m = length(gh.qn_du.memory_s)
    for i in m:-1:1
        # what if division by zero happened here, setting to zero ignores this in the next step
        # pre-compute in case inner is expensive
        v = inner(M, p, gh.qn_du.memory_s[i], gh.qn_du.memory_y[i])
        if isnan(v)
            println("NaN in memory")
            @show gh.qn_du.memory_s[i]
            @show gh.qn_du.memory_y[i]
        end
        if v < gh.iszero_abstol
            # The inner products ⟨s_i,y_i⟩ ≈ 0, i=$i, ignoring summand in approximation.
            # (s, y) pairs with negative inner product are broken anyway, so we can reject them here
            gh.ρ[i] = zero(eltype(gh.ρ))
        else
            gh.ρ[i] = 1 / v
            # it's so close to zero that we can skip it
            if abs(gh.ρ[i]) < gh.iszero_abstol
                gh.ρ[i] = zero(eltype(gh.ρ))
            end
        end
    end
    last_safe_index = -1
    for i in eachindex(gh.ρ)
        if abs(gh.ρ[i]) > 0
            last_safe_index = i
        end
    end

    if (last_safe_index == -1)
        # All memory yield zero inner products
        gh.current_scale = gh.initial_scale
        gh.M_11 = fill(0.0, 0, 0)
        gh.M_21 = fill(0.0, 0, 0)
        gh.M_22 = fill(0.0, 0, 0)
        return gh
    end

    invA = Diagonal([-ri for ri in gh.ρ if !iszero(ri)])
    num_nonzero_rho = count(!iszero, gh.ρ)

    Lk = LowerTriangular(zeros(num_nonzero_rho, num_nonzero_rho))

    # total scaling factor for the initial Hessian
    gh.current_scale = (gh.ρ[last_safe_index] * norm(M, p, gh.qn_du.memory_y[last_safe_index])^2) / gh.initial_scale

    tsksk = Symmetric(zeros(num_nonzero_rho, num_nonzero_rho))
    ii = 1
    # fill Dk and Lk
    for i in 1:m
        if iszero(gh.ρ[i])
            continue
        end
        jj = 1
        for j in 1:m
            if iszero(gh.ρ[j])
                continue
            end
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

function hess_val_single_pass(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, X)
    M = gh.M
    p = gh.p
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.ρ)

    normX_sqr = norm(M, p, X)^2

    if m == 0 || num_nonzero_rho == 0
        return gh.initial_scale \ normX_sqr
    end

    ii = 1
    for i in 1:m
        if iszero(gh.ρ[i])
            continue
        end
        gh.coords_Yk_X[ii] = inner(M, p, gh.qn_du.memory_y[i], X)
        gh.coords_Sk_X[ii] = gh.current_scale * inner(M, p, gh.qn_du.memory_s[i], X)

        ii += 1
    end
    coords_Yk = view(gh.coords_Yk_X, 1:num_nonzero_rho)
    coords_Sk = view(gh.coords_Sk_X, 1:num_nonzero_rho)

    return hess_val_from_wmwt_coords(gh, normX_sqr, coords_Yk, coords_Sk, coords_Yk, coords_Sk)
end

function hess_val_single_pass_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b)
    M = gh.M
    p = gh.p
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.ρ)

    if m == 0 || num_nonzero_rho == 0
        return inv(gh.initial_scale)
    end

    ii = 1
    for i in 1:m
        if iszero(gh.ρ[i])
            continue
        end
        gh.coords_Yk_X[ii] = get_at_bound_index(M, gh.qn_du.memory_y[i], b)
        gh.coords_Sk_X[ii] = gh.current_scale * get_at_bound_index(M, gh.qn_du.memory_s[i], b)

        ii += 1
    end
    coords_Yk = view(gh.coords_Yk_X, 1:num_nonzero_rho)
    coords_Sk = view(gh.coords_Sk_X, 1:num_nonzero_rho)

    return hess_val_from_wmwt_coords(gh, one(eltype(gh.ρ)), coords_Yk, coords_Sk, coords_Yk, coords_Sk)
end

function hess_val_single_pass_eb(gh::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b, Y)
    M = gh.M
    p = gh.p
    m = length(gh.qn_du.memory_s)
    num_nonzero_rho = count(!iszero, gh.ρ)

    Yb = get_at_bound_index(M, Y, b)
    if m == 0 || num_nonzero_rho == 0
        return gh.initial_scale * Yb
    end

    ii = 1
    for i in 1:m
        if iszero(gh.ρ[i])
            continue
        end
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
    set_M_current_scale!(gh)
    return gh
end


"""
    abstract type AbstractFPFPPUpdater end

Abstract type for methods that calculate f' and f'' in the GCP calculation in subsequent
line segments in `GCPFinder`.
"""
abstract type AbstractFPFPPUpdater end

"""
    init_updater!(::AbstractManifold, fpfpp_upd::AbstractFPFPPUpdater, d, ha)

Method for initialization of `AbstractFPFPPUpdater` `fpfpp_upd` just before the loop
that examines subsequent intervals for GCP. By default it does nothing.
"""
init_updater!(::AbstractManifold, fpfpp_upd::AbstractFPFPPUpdater, d, ha) = fpfpp_upd

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
    return LimitedMemoryFPFPPUpdater(similar(ha.ρ), similar(ha.ρ), similar(ha.ρ), similar(ha.ρ))
end

function (::GenericFPFPPUpdater)(M::AbstractManifold, old_f_prime, old_f_double_prime, dt, db, gb, ha, b, z, d_old)
    f_prime = old_f_prime + dt * old_f_double_prime - db * (gb + hess_val_eb(ha, b, z))
    f_double_prime = old_f_double_prime + (2 * -db * hess_val_eb(ha, b, d_old)) + db^2 * hess_val_eb(ha, b)

    return f_prime, f_double_prime
end

function init_updater!(M::AbstractManifold, fpfpp_upd::LimitedMemoryFPFPPUpdater, d, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate)
    fpfpp_upd.c_s .= 0
    fpfpp_upd.c_y .= 0
    ii = 1
    for i in eachindex(ha.ρ)
        if iszero(ha.ρ[i])
            continue
        end

        fpfpp_upd.p_s[ii] = ha.current_scale * inner(M, ha.p, ha.memory_s[i], d)
        fpfpp_upd.p_y[ii] = inner(M, ha.p, ha.memory_y[i], d)
        ii += 1
    end
    return fpfpp_upd
end

function (fpfpp_upd::LimitedMemoryFPFPPUpdater)(M::AbstractManifold, old_f_prime, old_f_double_prime, dt, db, gb, ha::QuasiNewtonLimitedMemoryBoxDirectionUpdate, b, z, d_old)

    m = length(ha.memory_s)
    num_nonzero_rho = count(!iszero, ha.ρ)

    iss_eb_z = get_at_bound_index(M, z, b)
    iss_eb_d = get_at_bound_index(M, d_old, b)

    ii = 1
    for i in 1:m
        if iszero(ha.ρ[i])
            continue
        end
        # setting _X to w_b from the paper
        ha.coords_Yk_X[ii] = get_at_bound_index(M, ha.memory_y[i], b)
        ha.coords_Sk_X[ii] = ha.current_scale * get_at_bound_index(M, ha.memory_s[i], b)

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

    f_double_prime = old_f_double_prime - 2 * db * eb_B_d + db^2 * hess_val_eb(ha, b)

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


struct GCPFinder{TM <: AbstractManifold, TX, THA, TFU <: AbstractFPFPPUpdater}
    M::TM
    Y_tmp::TX
    d_old::TX
    ha::THA
    fpfpp_updater::TFU
end

function GCPFinder(
        M::AbstractManifold, p, ha::AbstractQuasiNewtonDirectionUpdate;
        fpfpp_updater::AbstractFPFPPUpdater = get_default_fpfpp_updater(ha)
    )
    return GCPFinder(M, zero_vector(M, p), zero_vector(M, p), ha, fpfpp_updater)
end

"""
    find_gcp!(gcp::GCPFinder, p_cp, p, d, X)

Find generalized Cauchy point looking from point `p` in direction `d` and save it to `p_cp`.
Gradient of the objective at `p` is `X`.

The function returns `true` if the point was found and `false` otherwise.
"""
function find_gcp!(gcp::GCPFinder, p_cp, p, d, X)
    M = gcp.M
    copyto!(M, p_cp, p)
    zero_vector!(M, gcp.Y_tmp, p)

    bounds_indices = get_bounds_index(M)
    TInd = eltype(bounds_indices)

    t = Dict{TInd, Float64}((k, Inf) for k in bounds_indices)

    F_list = Tuple{Float64, TInd}[]
    sizehint!(F_list, length(bounds_indices) + 1)

    for i in bounds_indices
        t[i] = get_bound_t(M, p, d, i)

        if t[i] > 0
            push!(F_list, (t[i], i))
        end
    end

    if isempty(F_list)
        @warn "We can't go in the selected direction"
        return false
    end

    if M isa ProductManifold
        # push also `t` corresponding to max_stepsize if it is considered in the manifold
        M2 = M.manifolds[2]
        p2 = submanifold_component(M, p, Val(2))
        max_step = Manopt.max_stepsize(M2, p2)
        if isfinite(max_step)
            d2 = submanifold_component(M, d, Val(2))
            tms = max_step / norm(M2, p2, d2)
            push!(F_list, (tms, -1))
        end
    end

    F = BinaryHeap(Base.By(first), F_list)

    f_prime = inner(M, p, X, d)
    f_double_prime = hess_val(gcp.ha, d)

    if iszero(f_prime) || iszero(f_double_prime)
        return false
    end

    dt_min = -f_prime / f_double_prime
    t_old = 0.0

    t_current, b = pop!(F)
    dt = t_current - t_old

    init_updater!(M, gcp.fpfpp_upd, d, gcp.ha)
    # b can be -1 if it corresponds to the max stepsize limit on the manifold part
    while dt_min > dt && b != -1
        gcp.Y_tmp .+= dt .* d
        copyto!(M, gcp.d_old, d)
        set_bound_at_index!(M, p_cp, d, b)

        gb = get_at_bound_index(M, grad, b)
        db = get_at_bound_index(M, gcp.d_old, b)

        f_prime, f_double_prime = gcp.fpfpp_upd(M, f_prime, f_double_prime, dt, db, gb, gcp.ha, b, gcp.Y_tmp, gcp.d_old)
        t_old = t_current

        # If f_prime is 0, we've found the local minimizer (GCP)
        if iszero(f_prime) || iszero(f_double_prime)
            # It means that GCP is at the beginning of the t_current, so we want to set dt_min to 0 (stay in the point)
            dt_min = 0.0
            break
        end

        dt_min = -f_prime / f_double_prime

        if isempty(F)
            break
        end

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

    return true
end
