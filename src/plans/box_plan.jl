

mutable struct QuasiNewtonLimitedMemoryBoxDirectionUpdate{
        TDU <: QuasiNewtonLimitedMemoryDirectionUpdate
    } <: AbstractQuasiNewtonDirectionUpdate
    qn_du::TDU
end

abstract type AbstractFPFPPUpdater end

init_updater!(::AbstractManifold, fpfpp_upd::AbstractFPFPPUpdater, d, ha) = fpfpp_upd

struct GenericFPFPPUpdater <: AbstractFPFPPUpdater end

get_default_fpfpp_updater(::MatrixHessianApproximation) = GenericFPFPPUpdater()

struct LimitedMemoryFPFPPUpdater{TV<:AbstractVector} <: AbstractFPFPPUpdater
    p_s::TV
    p_y::TV
    c_s::TV
    c_y::TV
end

function get_default_fpfpp_updater(ha::LimitedMemoryHessianApproximation)
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

function (fpfpp_upd::LimitedMemoryFPFPPUpdater)(M::AbstractManifold, old_f_prime, old_f_double_prime, dt, db, gb, ha::LimitedMemoryHessianApproximation, b, z, d_old)

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
    get_bounds_index(::HyperrectangleProduct)

Get the bound indices of manifold `M`. Standard manifolds don't have bounds, so
`Base.OneTo(1)` is returned.
"""
get_bounds_index(M::Hyperrectangle) = eachindex(M.lb)
get_bounds_index(M::ProductManifold) = get_bounds_index(M.manifolds[1])

"""
    get_bound_t(M::AbstractManifold, x, d, i)

Get the upper bound on moving in direction `d` from point `p` on manifold `M`, for the
bound index `i`.
"""
function get_bound_t(M::Hyperrectangle, p, d, i)
    if d[i] > 0
        return (M.ub[i] - p[i]) / d[i]
    elseif d[i] < 0
        return (M.lb[i] - p[i]) / d[i]
    else
        return Inf
    end
end
function get_bound_t(M::ProductManifold, p, d, i)
    return get_bound_t(M.manifolds[1], submanifold_component(M, p, Val(1)), submanifold_component(M, d, Val(1)), i)
end
function set_bound_t_at_index!(M::ProductManifold, p_cp, t, d, i)
    set_bound_t_at_index!(M.manifolds[1], submanifold_component(M, p_cp, Val(1)), t, d, i)
end

function set_bound_t_at_index!(::Hyperrectangle, p_cp, t, d, i)
    p_cp[i] += t * d[i]
end

function set_bound_at_index!(M::Hyperrectangle, p_cp, d, i)
    p_cp[i] = d[i] > 0 ? M.ub[i] : M.lb[i]
    d[i] = 0
end

function set_bound_at_index!(M::ProductManifold, p_cp, d, i)
    set_bound_at_index!(M.manifolds[1], submanifold_component(M, p_cp, Val(1)), submanifold_component(M, d, Val(1)), i)
end


struct GCPFinder{TM<:AbstractManifold,TX,THA,TFU<:AbstractFPFPPUpdater}
    M::TM
    Y_tmp::TX
    d_old::TX
    ha::THA
    fpfpp_updater::TFU
end

function GCPFinder(M::AbstractManifold, p, ha; fpfpp_updater=get_default_fpfpp_updater(ha))
    return GCPFinder(M, zero_vector(M, p), zero_vector(M, p), ha, fpfpp_updater)
end

"""
    find_gcp!(gcp::GCPFinder, p_cp, p, d, X, ha)

Find generalized Cauchy point looking from point `p` in direction `d` and save it to `p_cp`.
Gradient of the objective at `p` is `X`.
"""
function find_gcp!(gcp::GCPFinder, p_cp, p, d, X, ha)

    M = gcp.M
    copyto!(M, p_cp, p)
    zero_vector!(M, gcp.Y_tmp, p)
    
    bounds_indices = get_bounds_index(M)
    TInd = eltype(bounds_indices)

    t = Dict{TInd,Float64}((k, Inf) for k in bounds_indices)

    F_list = Tuple{Float64,TInd}[]
    sizehint!(F_list, length(bounds_indices)+1)

    for i in bounds_indices
        t[i] = get_bound_t(M, p, d, i)

        if t[i] > 0
            push!(F_list, (t[i], i))
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
    end

    if isempty(F_list)
        @warn "We can't go in the selected direction"
        return false
    end

    F = BinaryHeap(Base.By(first), F_list)

    f_prime = inner(M, p, X, d)
    f_double_prime = hess_val(ha, d)

    if iszero(f_prime) || iszero(f_double_prime)
        return false
    end

    dt_min = -f_prime / f_double_prime
    t_old = 0.0

    t_current, b = pop!(F)
    dt = t_current - t_old
    
    init_updater!(M, gcp.fpfpp_upd, d, ha)
    # b can be -1 if it corresponds to the max stepsize limit on the manifold part
    while dt_min > dt && b != -1
        gcp.Y_tmp .+= dt .* d
        copyto!(M, gcp.d_old, d)
        set_bound_at_index!(M, p_cp, d, b)

        gb = get_at_bound_index(M, grad, b)
        db = get_at_bound_index(M, gcp.d_old, b)
     
        f_prime, f_double_prime = gcp.fpfpp_upd(M, f_prime, f_double_prime, dt, db, gb, ha, b, gcp.Y_tmp, gcp.d_old)
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

