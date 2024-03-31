#
# State
#
@doc raw"""
    CMAESState{P,T} <: AbstractManoptSolverState

State of covariance matrix adaptation evolution strategy.
"""
mutable struct CMAESState{
    P,
    TParams<:Real,
    TStopping<:StoppingCriterion,
    TRetraction<:AbstractRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
    TB<:AbstractBasis,
    TRng<:AbstractRNG,
} <: AbstractManoptSolverState
    p::P # best point found so far
    p_obj::TParams
    μ::Int # parent number
    λ::Int # population size
    μ_eff::TParams # variance effective selection mass for the mean
    c_1::TParams # learning rate for the rank-one update
    c_c::TParams # decay rate for cumulation path for the rank-one update
    c_μ::TParams # learning rate for the rank-μ update
    c_σ::TParams # decay rate for the comulation path for the step-size control
    c_m::TParams # learning rate for the mean
    d_σ::TParams # damping parameter for step-size update
    stop::TStopping
    population::Vector{P} # population of the current generation
    ys_c::Vector{Vector{TParams}}
    covm::Matrix{TParams} # coordinates of the covariance matrix
    covm_cond::TParams # condition number of covm, updated after eigendecomposition
    best_fitness_current_gen::TParams
    median_fitness_current_gen::TParams
    worst_fitness_current_gen::TParams
    p_m::P # point around which we search for new candidates
    σ::TParams # step size
    p_σ::Vector{TParams} # coordinates of a vector in T_{p_m} M
    p_c::Vector{TParams} # coordinates of a vector in T_{p_m} M
    deviations::Vector{TParams} # standard deviations of coordinate RNG
    e_mv_norm::TParams # expected value of norm of the n-variable standard normal distribution
    recombination_weights::Vector{TParams}
    retraction_method::TRetraction
    vector_transport_method::TVTM
    basis::TB
    rng::TRng
end
function show(io::IO, s::CMAESState)
    i = get_count(s, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(s.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Covariance Matrix Adaptation Evolutionary Strategy
    $Iter
    ## Parameters
    * μ:                         $(s.μ)
    * λ:                         $(s.λ)
    * μ_eff:                     $(s.μ_eff)
    * c_1:                       $(s.c_1)
    * c_c:                       $(s.c_c)
    * c_μ:                       $(s.c_μ)
    * c_σ:                       $(s.c_σ)
    * c_m:                       $(s.c_m)
    * d_σ:                       $(s.d_σ)
    * recombination_weights:     $(s.recombination_weights)
    * retraction method:         $(s.retraction_method)
    * vector transport method:   $(s.vector_transport_method)
    * basis:                     $(s.basis)

    ## Current values
    * covm:                       $(s.covm)
    * covm_cond:                  $(s.covm_cond)
    * best_fitness_current_gen:   $(s.best_fitness_current_gen)
    * median_fitness_current_gen: $(s.median_fitness_current_gen)
    * worst_fitness_current_gen:  $(s.worst_fitness_current_gen)
    * σ:                          $(s.σ)
    * p_σ:                        $(s.p_σ)
    * p_c:                        $(s.p_c)
    * p_m:                        $(s.p_m)
    * deviations:                 $(s.deviations)

    ## Stopping criterion

    $(status_summary(s.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
#
# Access functions
#
get_iterate(pss::CMAESState) = pss.p

function initialize_solver!(mp::AbstractManoptProblem, s::CMAESState)
    M = get_manifold(mp)
    n_coords = number_of_coordinates(M, s.basis)
    s.covm = Matrix{number_eltype(s.p)}(I, n_coords, n_coords)
    s.covm_cond = 1

    return s
end
function step_solver!(mp::AbstractManoptProblem, s::CMAESState, iteration::Int)
    M = get_manifold(mp)
    n_coords = number_of_coordinates(M, s.basis)

    # sampling and evaluation of new solutions

    D2, B = eigen(Symmetric(s.covm))
    min_eigval, max_eigval = extrema(abs.(D2))
    s.covm_cond = max_eigval / min_eigval
    s.deviations .= sqrt.(D2)
    cov_invsqrt = B * Diagonal(inv.(s.deviations)) * B'
    Y_m = zero_vector(M, s.p_m)
    for i in 1:(s.λ)
        mul!(s.ys_c[i], B, s.deviations .* randn(s.rng, n_coords)) # Eqs. (38) and (39)
        get_vector!(M, Y_m, s.p_m, s.ys_c[i], s.basis) # Eqs. (38) and (39)
        retract!(M, s.population[i], s.p_m, Y_m, s.σ, s.retraction_method) # Eq. (40)
    end
    fitness_vals = map(p -> get_cost(mp, p), s.population)
    s.best_fitness_current_gen, s.worst_fitness_current_gen = extrema(fitness_vals)
    s.median_fitness_current_gen = median(fitness_vals)
    for (i, fitness) in enumerate(fitness_vals)
        if fitness < s.p_obj
            s.p_obj = fitness
            copyto!(s.p, s.population[i])
        end
    end

    # sorting solutions
    ys_c_sorted = map(x -> x[1], sort(collect(zip(s.ys_c, fitness_vals)); by=f -> f[2]))

    # selection and recombination
    wmean_y_c = sum(s.recombination_weights[1:(s.μ)] .* ys_c_sorted[1:(s.μ)]) # Eq. (41)
    new_m = retract(
        M, s.p_m, get_vector(M, s.p_m, wmean_y_c, s.basis), s.c_m * s.σ, s.retraction_method
    ) # Eq. (42)

    # step-size control
    cinv_y = (cov_invsqrt * wmean_y_c)
    s.p_σ .= (1 - s.c_σ) * s.p_σ + sqrt(s.c_σ * (2 - s.c_σ) * s.μ_eff) * cinv_y # Eq. (43)
    s.σ *= exp(s.c_σ / s.d_σ * ((norm(s.p_σ) / s.e_mv_norm) - 1)) # Eq. (44)

    # covariance matrix adaptation
    s.p_c .*= 1 - s.c_c # Eq. (45), part 1
    if norm(s.p_σ) / sqrt(1 - (1 - s.c_σ)^(2 * (iteration + 1))) <
        (1.4 + 2 / (n_coords + 1)) * s.e_mv_norm # h_σ criterion
        s.p_c .+= sqrt(s.c_c * (2 - s.c_c) * s.μ_eff) .* wmean_y_c # Eq. (45), part 2
        δh_σ = zero(s.c_c) # Appendix A
    else
        δh_σ = s.c_c * (2 - s.c_c) # Appendix A
    end
    s.covm .*= (1 + s.c_1 * δh_σ - s.c_1 - s.c_μ * sum(s.recombination_weights)) # Eq. (47), part 1  
    mul!(s.covm, s.p_c, s.p_c', s.c_1, true) # Eq. (47), rank 1 update
    for i in 1:(s.λ)
        w_i = s.recombination_weights[i]
        wᵒi = w_i # Eq. (46)
        if w_i < 0
            mul!(cinv_y, cov_invsqrt, s.ys_c[i])
            wᵒi *= n_coords / norm(cinv_y)^2
        end
        mul!(s.covm, s.ys_c[i], s.ys_c[i]', s.c_μ * wᵒi, true) # Eq. (47), rank μ update
    end
    # move covariance matrix, p_c and p_σ to new mean point
    s.covm .= spd_matrix_transport_to(
        M, s.p_m, s.covm, new_m, s.basis, s.vector_transport_method
    )
    get_vector!(M, Y_m, s.p_m, s.p_σ, s.basis)
    vector_transport_to!(M, Y_m, s.p_m, Y_m, new_m, s.vector_transport_method)
    get_coordinates!(M, s.p_σ, new_m, Y_m, s.basis)

    get_vector!(M, Y_m, s.p_m, s.p_c, s.basis)
    vector_transport_to!(M, Y_m, s.p_m, Y_m, new_m, s.vector_transport_method)
    get_coordinates!(M, s.p_c, new_m, Y_m, s.basis)

    # update mean point
    copyto!(M, s.p_m, new_m)
    return s
end

function cma_es(M::AbstractManifold, f, p_m=rand(M); kwargs...)
    mco = ManifoldCostObjective(f)
    return cma_es(M, mco, p_m; kwargs...)
end

function default_cma_es_stopping_criterion(
    M::AbstractManifold, λ::Int; tol_fun::TParam=1e-12, tol_x::TParam=1e-12
) where {TParam<:Real}
    return StopAfterIteration(50000) |
           CMAESConditionCov() |
           EqualFunValuesCondition{TParam}(Int(10 + ceil(30 * manifold_dimension(M) / λ))) |
           StagnationCondition(
               Int(120 + 30 * ceil(30 * manifold_dimension(M) / λ)), 20000, 0.3
           ) |
           TolXUpCondition(1e4) |
           TolFunCondition(tol_fun, Int(10 + ceil(30 * manifold_dimension(M) / λ))) |
           TolXCondition(tol_x)
end

@doc raw"""
    cma_es(M, f, p_m=rand(M); σ::Real=1.0, kwargs...)

Perform covariance matrix adaptation evolutionary strategy search for global gradient-free
randomized optimization. It is suitable for complicated non-convex functions. It can be
reasonably expected to find global minimum within 3σ distance from `p_m`.

Implementation is based on [Hansen:2023](@cite) with basic adaptations to the Riemannian
setting.

# Input

* `M`:      a manifold ``\mathcal M``
* `f`:      a cost function ``f: \mathcal M→ℝ`` to find a minimizer ``p^*`` for

# Optional

* `p_m`:    (`rand(M)`) an initial point `p`
* `σ`       (`1.0`) initial standard deviation
* `λ`       (`4 + Int(floor(3 * log(manifold_dimension(M))))`population size (can be
  increased for a more thorough search)
* `tol_fun` (`1e-12`) tolerance for the `TolFunCondition`, similar to absolute difference
  between function values at subsequent points
* `tol_x`   (`1e-12`) tolerance for the `TolXCondition`, similar to absolute difference
  between subsequent point but actually computed from distribution parameters.

# Output

the obtained (approximate) minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details.
"""
function cma_es(
    M::AbstractManifold,
    mco::O,
    p_m=rand(M);
    σ::Real=1.0,
    λ::Int=4 + Int(floor(3 * log(manifold_dimension(M)))), # Eq. (48)
    tol_fun::Real=1e-12,
    tol_x::Real=1e-12,
    stopping_criterion::StoppingCriterion=default_cma_es_stopping_criterion(
        M, λ; tol_fun=tol_fun, tol_x=tol_x
    ),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p_m)),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M, typeof(p_m)
    ),
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    rng::AbstractRNG=default_rng(),
    kwargs..., #collect rest
) where {O<:Union{AbstractManifoldCostObjective,AbstractDecoratedManifoldObjective}}
    dmco = decorate_objective!(M, mco; kwargs...)
    mp = DefaultManoptProblem(M, dmco)
    n_coords = number_of_coordinates(M, basis)
    wp = [log((λ + 1) / 2) - log(i) for i in 1:λ] # Eq. (49)
    μ = Int(floor(λ / 2)) # Table 1 caption
    μ_eff = (sum(wp[1:μ])^2) / (sum(x -> x^2, wp[1:μ]))  # Table 1 caption
    @assert μ_eff >= 1
    @assert μ_eff <= μ
    μ_eff⁻ = (sum(wp[(μ + 1):end])^2) / (sum(x -> x^2, wp[(μ + 1):end]))

    αμ_eff⁻ = 1 + (2 * μ_eff⁻) / (μ_eff + 2) # Eq. (51)
    α_cov = 2.0 # Eq. (57)
    c_1 = α_cov / ((n_coords + 1.3)^2 + μ_eff) # Eq. (57)
    c_μ = min(
        1 - c_1,
        α_cov * (0.25 + μ_eff + 1 / μ_eff - 2) / ((n_coords + 2)^2 + α_cov * μ_eff / 2),
    ) # Eq. (58)

    αμ⁻ = 1 + c_1 / c_μ # Eq. (50)
    α_posdef⁻ = (1 - c_1 - c_μ) / (n_coords * c_μ) # Eq. (52)
    w_normalization_positive = 1 / sum([wj for wj in wp if wj > 0]) # Eq. (53)
    w_normalization_negative =
        -min(αμ⁻, αμ_eff⁻, α_posdef⁻) / sum([wj for wj in wp if wj < 0]) # Eq. (53)
    recombination_weights = [
        w_i * (w_i >= 0 ? w_normalization_positive : w_normalization_negative) for w_i in wp
    ] # Eq. (53)
    c_m = 1.0 # Note below Eq. (9)

    @assert sum(recombination_weights[1:μ]) ≈ 1
    c_σ = (μ_eff + 2) / (n_coords + μ_eff + 5) # Eq. (55)
    d_σ = 1 + 2 * max(0, sqrt((μ_eff - 1) / (n_coords + 1)) - 1) + c_σ # Eq. (55)
    c_c = (4 + μ_eff / n_coords) / (n_coords + 4 + 2 * μ_eff / n_coords) # Eq. (56)
    population = [allocate(M, p_m) for _ in 1:λ]
    # approximation of expected value of norm of standard n_coords-variate normal distribution
    e_mv_norm = sqrt(n_coords) * (1 - 1 / (4 * n_coords) + 1 / (21 * n_coords))
    state = CMAESState(
        allocate(M, p_m),
        Inf,
        μ,
        λ,
        μ_eff,
        c_1,
        c_c,
        c_μ,
        c_σ,
        c_m,
        d_σ,
        stopping_criterion,
        population,
        [similar(Vector{Float64}, n_coords) for _ in 1:λ],
        Matrix{number_eltype(p_m)}(I, n_coords, n_coords),
        one(c_1),
        Inf,
        Inf,
        Inf,
        p_m,
        σ,
        zeros(typeof(c_1), n_coords),
        zeros(typeof(c_1), n_coords),
        ones(typeof(c_1), n_coords),
        e_mv_norm,
        recombination_weights,
        retraction_method,
        vector_transport_method,
        basis,
        rng,
    )

    d_state = decorate_state!(state; kwargs...)
    solve!(mp, d_state)
    return get_solver_return(get_objective(mp), d_state)
end

@doc raw"""
    spd_matrix_transport_to(M::AbstractManifold, p, spd_coords, q, basis::AbstractBasis, vtm::AbstractVectorTransportMethod)

Transport the SPD matrix with `spd_coords` when expanded in `basis` from point `p` to
point `q` on `M`.

`(p, spd_coords)` belongs to the fiber bundle of ``B = \mathcal M × SPD(n)``, where `n`
is the (real) dimension of `M`. The function corresponds to the Ehresmann connection
defined by vector transport `vtm` of eigenvectors of `spd_coords`.
"""
function spd_matrix_transport_to(
    M::AbstractManifold,
    p,
    spd_coords,
    q,
    basis::AbstractBasis,
    vtm::AbstractVectorTransportMethod,
)
    if is_flat(M)
        return spd_coords
    end
    D, Q = eigen(Symmetric(spd_coords))
    n = length(D)
    vectors = [get_vector(M, p, Q[:, i], basis) for i in 1:n]
    for i in 1:n
        vector_transport_to!(M, vectors[i], p, vectors[i], q, vtm)
    end
    coords = [get_coordinates(M, p, vectors[i], basis) for i in 1:n]
    Qt = reduce(hcat, coords)
    return Qt * Diagonal(D) * Qt'
end

"""
    CMAESConditionCov <: StoppingCriterion

Stop CMA-ES if condition number of covariance matrix exceeds `threshold`.
"""
mutable struct CMAESConditionCov{T<:Real} <: StoppingCriterion
    threshold::T
    last_cond::T
    at_iteration::Int
end
function CMAESConditionCov(threshold::Real=1e14)
    return CMAESConditionCov{typeof(threshold)}(threshold, 1, 0)
end

indicates_convergence(c::CMAESConditionCov) = false
is_active_stopping_criterion(c::CMAESConditionCov) = c.at_iteration > 0
function (c::CMAESConditionCov)(::AbstractManoptProblem, s::CMAESState, i::Int)
    if i == 0 # reset on init
        c.at_iteration = 0
        return false
    end
    c.last_cond = s.covm_cond
    if i > 0 && c.last_cond > c.threshold
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::CMAESConditionCov)
    has_stopped = c.at_iteration > 0
    s = has_stopped ? "reached" : "not reached"
    return "cond(s.covm) > $(c.threshold):\t$s"
end
function get_reason(c::CMAESConditionCov)
    return "At iteration $(c.at_iteration) the condition number of covariance matrix ($(c.last_cond)) exceeded the threshold ($(c.threshold)).\n"
end
function show(io::IO, c::CMAESConditionCov)
    return print(io, "CMAESConditionCov($(c.threshold))\n    $(status_summary(c))")
end

"""
    EqualFunValuesCondition <: StoppingCriterion

Stop if the range of the best objective function values of the last `iteration_range`
generations is zero.

See also `TolFunCondition`.
"""
mutable struct EqualFunValuesCondition{TParam<:Real} <: StoppingCriterion
    iteration_range::Int
    best_objective_at_last_change::TParam
    iterations_since_change::Int
end

function EqualFunValuesCondition{TParam}(iteration_range::Int) where {TParam}
    return EqualFunValuesCondition{TParam}(iteration_range, Inf, 0)
end

# It just indicates stagnation, not that we converged to a minimizer
indicates_convergence(c::EqualFunValuesCondition) = true
function is_active_stopping_criterion(c::EqualFunValuesCondition)
    return c.iterations_since_change >= c.iteration_range
end
function (c::EqualFunValuesCondition)(::AbstractManoptProblem, s::CMAESState, i::Int)
    if i == 0 # reset on init
        c.best_objective_at_last_change = Inf
        return false
    end
    if c.iterations_since_change >= c.iteration_range
        return true
    else
        if c.best_objective_at_last_change != s.best_fitness_current_gen
            c.best_objective_at_last_change = s.best_fitness_current_gen
            c.iterations_since_change = 0
        else
            c.iterations_since_change += 1
        end
    end
    return false
end
function status_summary(c::EqualFunValuesCondition)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "c.iterations_since_change > $(c.iteration_range):\t$s"
end
function get_reason(c::EqualFunValuesCondition)
    return "For the last $(c.iterations_since_change) generation the best objective value in each generation was equal to $(c.best_objective_at_last_change).\n"
end
function show(io::IO, c::EqualFunValuesCondition)
    return print(
        io, "EqualFunValuesCondition($(c.iteration_range))\n    $(status_summary(c))"
    )
end

"""
    StagnationCondition{TParam<:Real} <: StoppingCriterion

The best and median fitness in each iteraion is tracked over the last 20% but
at least `min_size` and no more than `max_size` iterations. Solver is stopped if
in both histories the median of the most recent `fraction` of values is not better
than the median of the oldest `fraction`.
"""
mutable struct StagnationCondition{TParam<:Real} <: StoppingCriterion
    min_size::Int
    max_size::Int
    fraction::TParam
    best_history::CircularBuffer{TParam}
    median_history::CircularBuffer{TParam}
end

function StagnationCondition(
    min_size::Int, max_size::Int, fraction::TParam
) where {TParam<:Real}
    return StagnationCondition{TParam}(
        min_size,
        max_size,
        fraction,
        CircularBuffer{TParam}(max_size),
        CircularBuffer{TParam}(max_size),
    )
end

# It just indicates stagnation, not that we converged to a minimizer
indicates_convergence(c::StagnationCondition) = true
function is_active_stopping_criterion(c::StagnationCondition)
    N = length(c.best_history)
    if N < c.min_size
        return false
    end
    thr_low = Int(ceil(N * c.fraction))
    thr_high = Int(floor(N * (1 - c.fraction)))
    best_stagnant =
        median(c.best_history[1:thr_low]) <= median(c.best_history[thr_high:end])
    median_stagnant =
        median(c.median_history[1:thr_low]) <= median(c.median_history[thr_high:end])
    return best_stagnant && median_stagnant
end
function (c::StagnationCondition)(::AbstractManoptProblem, s::CMAESState, i::Int)
    if i == 0 # reset on init
        empty!(c.best_history)
        empty!(c.median_history)
        return false
    end
    if is_active_stopping_criterion(c)
        return true
    else
        push!(c.best_history, s.best_fitness_current_gen)
        push!(c.median_history, s.median_fitness_current_gen)
    end
    return false
end
function status_summary(c::StagnationCondition)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    N = length(c.best_history)
    thr_low = Int(ceil(N * c.fraction))
    thr_high = Int(floor(N * (1 - c.fraction)))
    median_best_old = median(c.best_history[1:thr_low])
    median_best_new = median(c.best_history[thr_high:end])
    median_median_old = median(c.median_history[1:thr_low])
    median_median_new = median(c.median_history[thr_high:end])
    return "generation >= $(c.min_size) && $(median_best_old) <= $(median_best_new) && $(median_median_old) <= $(median_median_new):\t$s"
end
function get_reason(::StagnationCondition)
    return "Both median and best objective history became stagnant.\n"
end
function show(io::IO, c::StagnationCondition)
    return print(
        io,
        "StagnationCondition($(c.min_size), $(c.max_size), $(c.fraction))\n    $(status_summary(c))",
    )
end

"""
    TolXCondition{TParam<:Real} <: StoppingCriterion

Stop if the standard deviation in all coordinates is smaller than `tol` and
norm of `σ * p_c` is smaller than `tol`.
"""
mutable struct TolXCondition{TParam<:Real} <: StoppingCriterion
    tol::TParam
    is_active::Bool
end
TolXCondition(tol::Real) = TolXCondition{typeof(tol)}(tol, false)

# It just indicates stagnation, not that we converged to a minimizer
indicates_convergence(c::TolXCondition) = true
function is_active_stopping_criterion(c::TolXCondition)
    return c.is_active
end
function (c::TolXCondition)(::AbstractManoptProblem, s::CMAESState, i::Int)
    if i == 0 # reset on init
        c.is_active = false
        return false
    end
    norm_inf_dev = norm(s.deviations, Inf)
    norm_inf_p_c = norm(s.p_c, Inf)
    c.is_active = norm_inf_dev < c.tol && s.σ * norm_inf_p_c < c.tol
    return c.is_active
end
function status_summary(c::TolXCondition)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "norm(s.deviations, Inf) < $(c.tol) && norm(s.σ * s.p_c, Inf) < $(c.tol) :\t$s"
end
function get_reason(c::TolXCondition)
    return "Standard deviation in all coordinates is smaller than $(c.tol) and `σ * p_c` has Inf norm lower than $(c.tol).\n"
end
function show(io::IO, c::TolXCondition)
    return print(io, "TolXCondition($(c.tol))\n    $(status_summary(c))")
end

"""
    TolXUpCondition{TParam<:Real} <: StoppingCriterion

Stop if `σ` times maximum deviation increased by more than `tol`. This usually indicates a
far too small `σ`, or divergent behavior.
"""
mutable struct TolXUpCondition{TParam<:Real} <: StoppingCriterion
    tol::TParam
    last_σ_times_maxstddev::TParam
    is_active::Bool
end
TolXUpCondition(tol::Real) = TolXUpCondition{typeof(tol)}(tol, 1.0, false)

indicates_convergence(c::TolXUpCondition) = false
function is_active_stopping_criterion(c::TolXUpCondition)
    return c.is_active
end
function (c::TolXUpCondition)(::AbstractManoptProblem, s::CMAESState, i::Int)
    if i == 0 # reset on init
        c.is_active = false
        return false
    end
    cur_σ_times_maxstddev = s.σ * maximum(s.deviations)
    if cur_σ_times_maxstddev / c.last_σ_times_maxstddev > c.tol
        c.is_active = true
        return true
    end
    return false
end
function status_summary(c::TolXUpCondition)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "cur_σ_times_maxstddev / c.last_σ_times_maxstddev > $(c.tol) :\t$s"
end
function get_reason(c::TolXUpCondition)
    return "σ times maximum standard deviation exceeded $(c.tol). This indicates either much too small σ or divergent behavior.\n"
end
function show(io::IO, c::TolXUpCondition)
    return print(io, "TolXUpCondition($(c.tol))\n    $(status_summary(c))")
end

"""
    TolFunCondition{TParam<:Real} <: StoppingCriterion

Stop if the range of the best objective function value in the last `max_size` generations
and all function values in the current generation is below `tol`.

# Constructor

    TolFunCondition(tol::Real, max_size::Int)
"""
mutable struct TolFunCondition{TParam<:Real} <: StoppingCriterion
    tol::TParam
    best_value_history::CircularBuffer{TParam}
    is_active::Bool
end
function TolFunCondition(tol::TParam, max_size::Int) where {TParam<:Real}
    return TolFunCondition{TParam}(tol, CircularBuffer{TParam}(max_size), false)
end

# It just indicates stagnation, not that we converged to a minimizer
indicates_convergence(c::TolFunCondition) = true
function is_active_stopping_criterion(c::TolFunCondition)
    return c.is_active
end
function (c::TolFunCondition)(::AbstractManoptProblem, s::CMAESState, i::Int)
    if i == 0 # reset on init
        c.is_active = false
        return false
    end
    push!(c.best_value_history, s.best_fitness_current_gen)
    if isfull(c.best_value_history)
        min_hist, max_hist = extrema(c.best_value_history)
        if max_hist - min_hist < c.tol &&
            s.best_fitness_current_gen - s.worst_fitness_current_gen < c.tol
            c.is_active = true
            return true
        end
    end
    return false
end
function status_summary(c::TolFunCondition)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "range of best objective values in the last $(length(c.best_value_history)) generations and all objective values in the current one < $(c.tol) :\t$s"
end
function get_reason(c::TolFunCondition)
    return "Range of best objective function values in the last $(length(c.best_value_history)) gnerations and all values in the current generation is below $(c.tol)\n"
end
function show(io::IO, c::TolFunCondition)
    return print(io, "TolFunCondition($(c.tol))\n    $(status_summary(c))")
end
