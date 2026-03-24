"""
    UnitVector{TB}

A type representing a unit tangent vector on a `Hyperrectangle`-like manifold with corners,
or a product of it with a standard manifold.
The field `index` stores the index of the element equal to 1.
All other elements are equal to 0.
`its` stores the overall iterator over all bounds.
"""
struct UnitVector{TB}
    index::TB
end

"""
    has_anisotropic_max_stepsize(M::AbstractManifold)

Return `true` if `M` has `max_stepsize` that depends on the direction.
For example, if `M` is a `Hyperrectangle`-like manifold with corners, or a product of it
with a standard manifold. Otherwise return `false`.
"""
has_anisotropic_max_stepsize(::AbstractManifold) = false
has_anisotropic_max_stepsize(M::ProductManifold) = any(has_anisotropic_max_stepsize, M.manifolds)

"""
    abstract type AbstractSegmentHessianUpdater end

Abstract type for methods that calculate f' and f'' in the GCD calculation in subsequent
line segments in [`GeneralizedCauchyDirectionSubsolver`](@ref).
"""
abstract type AbstractSegmentHessianUpdater end

"""
    init_updater!(::AbstractManifold, hessian_segment_updater::AbstractSegmentHessianUpdater, p, d, ha)

Method for initialization of `AbstractSegmentHessianUpdater` `hessian_segment_updater` just before the loop
that examines subsequent intervals for GCD.
"""
init_updater!(::AbstractManifold, hessian_segment_updater::AbstractSegmentHessianUpdater, p, d, ha)

"""
    struct GenericSegmentHessianUpdater <: AbstractSegmentHessianUpdater end

Generic f' and f'' calculation that only relies on `hessian_value` but is relatively slow for
high-dimensional domains.
"""
struct GenericSegmentHessianUpdater{TX} <: AbstractSegmentHessianUpdater
    d_z::TX
    d_tmp::TX
end

function get_default_hessian_segment_updater(M::AbstractManifold, p, ::Any)
    return GenericSegmentHessianUpdater(zero_vector(M, p), zero_vector(M, p))
end

function init_updater!(M::AbstractManifold, hessian_segment_updater::GenericSegmentHessianUpdater, p, d, ha)
    zero_vector!(M, hessian_segment_updater.d_z, p)
    copyto!(M, hessian_segment_updater.d_tmp, d)
    return hessian_segment_updater
end

@doc raw"""
    (upd::GenericSegmentHessianUpdater)(M::AbstractManifold, p, t::Real, dt::Real, b, db, ha)

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

function get_at_bound_index(M::ProductManifold, X, b::Tuple{Int, Any})
    return get_at_bound_index(M.manifolds[b[1]], submanifold_component(M, X, b[1]), b[2])
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
    GeneralizedCauchyDirectionSubsolver{TM <: AbstractManifold, TP, T_HA, TFU <: AbstractSegmentHessianUpdater}

Helper container for generalized Cauchy direction search. Stores the manifold `M`, cached
original descent direction (`d_original`), the Hessian approximation (`ha`), and the
`hessian_segment_updater`, which computes certain values of the Hessian while advancing segments.
Instances are reused across segments during [`find_generalized_cauchy_direction!`](@ref) to
avoid allocations.
"""
struct GeneralizedCauchyDirectionSubsolver{
        TX,
        T_HA, TFU <: AbstractSegmentHessianUpdater, TFT <: Tuple{<:Real, Any}, TBI,
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
        M::AbstractManifold, p, ha;
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
            <:Any, <:Any,
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
