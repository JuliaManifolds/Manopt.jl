#
# State
#
@doc """
    CMAESState{P,T} <: AbstractManoptSolverState

State of covariance matrix adaptation evolution strategy.

# Fields

$(_var(:Field, :p; add=" storing the best point found so far"))
* `p_obj`                       objective value at `p`
* `μ`                           parent number
* `λ`                           population size
* `μ_eff`                       variance effective selection mass for the mean
* `c_1`                         learning rate for the rank-one update
* `c_c`                         decay rate for cumulation path for the rank-one update
* `c_μ`                         learning rate for the rank-μ update
* `c_σ`                         decay rate for the cumulation path for the step-size control
* `c_m`                         learning rate for the mean
* `d_σ`                         damping parameter for step-size update
* `population`                  population of the current generation
* `ys_c`                        coordinates of random vectors for the current generation
* `covariance_matrix`           coordinates of the covariance matrix
* `covariance_matrix_eigen`     eigen decomposition of `covariance_matrix`
* `covariance_matrix_cond`      condition number of `covariance_matrix`, updated after eigen decomposition
* `best_fitness_current_gen`    best fitness value of individuals in the current generation
* `median_fitness_current_gen`  median fitness value of individuals in the current generation
* `worst_fitness_current_gen`   worst fitness value of individuals in the current generation
* `p_m`                         point around which the search for new candidates is done
* `σ`                           step size
* `p_σ`                         coordinates of a vector in ``$(_math(:TpM; p="p_m"))``
* `p_c`                         coordinates of a vector in ``$(_math(:TpM; p="p_m"))``
* `deviations`                  standard deviations of coordinate RNG
* `buffer`                      buffer for random number generation and `wmean_y_c` of length `n_coords`
* `e_mv_norm`                   expected value of norm of the `n_coords`-variable standard normal distribution
* `recombination_weights`       recombination weights used for updating covariance matrix
$(_var(:Field, :retraction_method))
$(_var(:Field, :stopping_criterion, "stop"))
$(_var(:Field, :vector_transport_method))
* `basis`                       a real coefficient basis for covariance matrix
* `rng`                         RNG for generating new points

# Constructor

    CMAESState(
        M::AbstractManifold,
        p_m::P,
        μ::Int,
        λ::Int,
        μ_eff::TParams,
        c_1::TParams,
        c_c::TParams,
        c_μ::TParams,
        c_σ::TParams,
        c_m::TParams,
        d_σ::TParams,
        stop::TStopping,
        covariance_matrix::Matrix{TParams},
        σ::TParams,
        recombination_weights::Vector{TParams};
        retraction_method::TRetraction=default_retraction_method(M, typeof(p_m)),
        vector_transport_method::TVTM=default_vector_transport_method(M, typeof(p_m)),
        basis::TB=default_basis(M, typeof(p_m)),
        rng::TRng=default_rng(),
    ) where {
        P,
        TParams<:Real,
        TStopping<:StoppingCriterion,
        TRetraction<:AbstractRetractionMethod,
        TVTM<:AbstractVectorTransportMethod,
        TB<:AbstractBasis,
        TRng<:AbstractRNG,
    }

# See also

[`cma_es`](@ref)
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
    p::P
    p_obj::TParams
    μ::Int
    λ::Int
    μ_eff::TParams
    c_1::TParams
    c_c::TParams
    c_μ::TParams
    c_σ::TParams
    c_m::TParams
    d_σ::TParams
    stop::TStopping
    population::Vector{P}
    ys_c::Vector{Vector{TParams}}
    covariance_matrix::Matrix{TParams}
    last_variances::Vector{TParams}
    covariance_matrix_eigen::Eigen{TParams,TParams,Matrix{TParams},Vector{TParams}}
    covariance_matrix_cond::TParams
    best_fitness_current_gen::TParams
    median_fitness_current_gen::TParams
    worst_fitness_current_gen::TParams
    p_m::P
    σ::TParams
    p_σ::Vector{TParams}
    p_c::Vector{TParams}
    deviations::Vector{TParams}
    buffer::Vector{TParams}
    e_mv_norm::TParams
    recombination_weights::Vector{TParams}
    retraction_method::TRetraction
    vector_transport_method::TVTM
    basis::TB
    rng::TRng
end

function CMAESState(
    M::AbstractManifold,
    p_m::P,
    μ::Int,
    λ::Int,
    μ_eff::TParams,
    c_1::TParams,
    c_c::TParams,
    c_μ::TParams,
    c_σ::TParams,
    c_m::TParams,
    d_σ::TParams,
    stop::TStopping,
    covariance_matrix::Matrix{TParams},
    σ::TParams,
    recombination_weights::Vector{TParams};
    retraction_method::TRetraction=default_retraction_method(M, typeof(p_m)),
    vector_transport_method::TVTM=default_vector_transport_method(M, typeof(p_m)),
    basis::TB=default_basis(M, P),
    rng::TRng=default_rng(),
) where {
    P,
    TParams<:Real,
    TStopping<:StoppingCriterion,
    TRetraction<:AbstractRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
    TB<:AbstractBasis,
    TRng<:AbstractRNG,
}
    n_coords = number_of_coordinates(M, basis)
    # approximation of expected value of norm of standard n_coords-variate normal distribution
    e_mv_norm = sqrt(n_coords) * (1 - 1 / (4 * n_coords) + 1 / (21 * n_coords))

    @assert μ_eff >= 1
    @assert μ_eff <= μ
    @assert sum(recombination_weights[1:μ]) ≈ 1
    cov_eig = eigen(covariance_matrix)

    return CMAESState{P,TParams,TStopping,TRetraction,TVTM,TB,TRng}(
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
        stop,
        [allocate(M, p_m) for _ in 1:λ],
        [similar(Vector{Float64}, n_coords) for _ in 1:λ],
        covariance_matrix,
        copy(cov_eig.values),
        cov_eig,
        1.0,
        Inf,
        Inf,
        Inf,
        p_m,
        σ,
        zeros(TParams, n_coords),
        zeros(TParams, n_coords),
        ones(TParams, n_coords),
        zeros(TParams, n_coords),
        e_mv_norm,
        recombination_weights,
        retraction_method,
        vector_transport_method,
        basis,
        rng,
    )
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
    * p_obj:                      $(s.p_obj)
    * covariance_matrix_cond:     $(s.covariance_matrix_cond)
    * best_fitness_current_gen:   $(s.best_fitness_current_gen)
    * median_fitness_current_gen: $(s.median_fitness_current_gen)
    * worst_fitness_current_gen:  $(s.worst_fitness_current_gen)
    * σ:                          $(s.σ)

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
    s.covariance_matrix = Matrix{number_eltype(s.p)}(I, n_coords, n_coords)
    s.covariance_matrix_cond = 1
    s.covariance_matrix_eigen = eigen(Symmetric(s.covariance_matrix))
    return s
end
function step_solver!(mp::AbstractManoptProblem, s::CMAESState, iteration::Int)
    M = get_manifold(mp)
    n_coords = number_of_coordinates(M, s.basis)

    # sampling and evaluation of new solutions

    # `D2, B = eigen(Symmetric(s.covariance_matrix))``
    D2, B = s.covariance_matrix_eigen # assuming eigendecomposition has already been completed
    min_eigval, max_eigval = extrema(abs.(D2))
    if minimum(D2) <= 0
        @warn "Covariance matrix has nonpositive eigenvalues; try reformulating the objective, modifying stopping criteria or adjusting optimization parameters to avoid this."
        # replace nonpositive variances with last positive entries; this is the approach used by pycma
        nonpos_inds = D2 .<= 0
        D2[nonpos_inds] .= s.last_variances[nonpos_inds]
    end
    s.covariance_matrix_cond = max_eigval / min_eigval
    s.deviations .= sqrt.(D2)
    cov_invsqrt = B * Diagonal(inv.(s.deviations)) * B'
    Y_m = zero_vector(M, s.p_m)
    for i in 1:(s.λ)
        randn!(s.rng, s.buffer) # Eqs. (38) and (39)
        s.buffer .*= s.deviations # Eqs. (38) and (39)
        mul!(s.ys_c[i], B, s.buffer) # Eqs. (38) and (39)
        get_vector!(M, Y_m, s.p_m, s.ys_c[i], s.basis) # Eqs. (38) and (39)
        ManifoldsBase.retract_fused!(
            M, s.population[i], s.p_m, Y_m, s.σ, s.retraction_method
        ) # Eq. (40)
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
    fill!(s.buffer, 0) # from now on until the end of this method buffer is ⟨y⟩_w from Eq. (41)
    for i in 1:(s.μ) # Eq. (41)
        s.buffer .+= s.recombination_weights[i] .* ys_c_sorted[i]
    end
    new_m = ManifoldsBase.retract_fused(
        M, s.p_m, get_vector(M, s.p_m, s.buffer, s.basis), s.c_m * s.σ, s.retraction_method
    ) # Eq. (42)

    # step-size control
    cinv_y = (cov_invsqrt * s.buffer)
    s.p_σ .= (1 - s.c_σ) * s.p_σ + sqrt(s.c_σ * (2 - s.c_σ) * s.μ_eff) * cinv_y # Eq. (43)
    s.σ *= exp(s.c_σ / s.d_σ * ((norm(s.p_σ) / s.e_mv_norm) - 1)) # Eq. (44)

    # covariance matrix adaptation
    s.p_c .*= 1 - s.c_c # Eq. (45), part 1
    if norm(s.p_σ) / sqrt(1 - (1 - s.c_σ)^(2 * (iteration + 1))) <
        (1.4 + 2 / (n_coords + 1)) * s.e_mv_norm # h_σ criterion
        s.p_c .+= sqrt(s.c_c * (2 - s.c_c) * s.μ_eff) .* s.buffer # Eq. (45), part 2
        δh_σ = zero(s.c_c) # Appendix A
    else
        δh_σ = s.c_c * (2 - s.c_c) # Appendix A
    end
    s.covariance_matrix .*= (
        1 + s.c_1 * δh_σ - s.c_1 - s.c_μ * sum(s.recombination_weights)
    ) # Eq. (47), part 1
    mul!(s.covariance_matrix, s.p_c, s.p_c', s.c_1, true) # Eq. (47), rank 1 update
    for i in 1:(s.λ)
        w_i = s.recombination_weights[i]
        wᵒi = w_i # Eq. (46)
        if w_i < 0
            mul!(cinv_y, cov_invsqrt, s.ys_c[i])
            wᵒi *= n_coords / norm(cinv_y)^2
        end
        mul!(s.covariance_matrix, s.ys_c[i], s.ys_c[i]', s.c_μ * wᵒi, true) # Eq. (47), rank μ update
    end
    # move covariance matrix, `p_c`, and `p_σ` to new mean point
    s.last_variances .= D2
    s.covariance_matrix_eigen = eigen(Symmetric(s.covariance_matrix))
    eigenvector_transport!(
        M, s.covariance_matrix_eigen, s.p_m, new_m, s.basis, s.vector_transport_method
    )
    mul!(
        s.covariance_matrix,
        s.covariance_matrix_eigen.vectors,
        Diagonal(s.covariance_matrix_eigen.values) * s.covariance_matrix_eigen.vectors',
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

@doc """
    cma_es(M, f, p_m=rand(M); σ::Real=1.0, kwargs...)

Perform covariance matrix adaptation evolutionary strategy search for global gradient-free
randomized optimization. It is suitable for complicated non-convex functions. It can be
reasonably expected to find global minimum within 3σ distance from `p_m`.

Implementation is based on [Hansen:2023](@cite) with basic adaptations to the Riemannian
setting.

# Input

* `M`:      a manifold ``$(_math(:M))``
* `f`:      a cost function ``f: $(_math(:M))→ℝ`` to find a minimizer ``p^*`` for

# Keyword arguments

* `p_m=`$(Manopt._link(:rand)): an initial point `p`
* `σ=1.0`: initial standard deviation
* `λ`:                  (`4 + Int(floor(3 * log(manifold_dimension(M))))`population size (can be
  increased for a more thorough global search but decreasing is not recommended)
* `tol_fun=1e-12`: tolerance for the `StopWhenPopulationCostConcentrated`, similar to
  absolute difference between function values at subsequent points
* `tol_x=1e-12`: tolerance for the `StopWhenPopulationStronglyConcentrated`, similar to
  absolute difference between subsequent point but actually computed from distribution
  parameters.
$(_var(:Keyword, :stopping_criterion; default="`default_cma_es_stopping_criterion(M, λ; tol_fun=tol_fun, tol_x=tol_x)`"))
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :vector_transport_method))
* `basis`               (`DefaultOrthonormalBasis()`) basis used to represent covariance in
* `rng=default_rng()`: random number generator for generating new points
  on `M`

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""
function cma_es(M::AbstractManifold, f; kwargs...)
    mco = ManifoldCostObjective(f)
    return cma_es!(M, mco, rand(M); kwargs...)
end
function cma_es(M::AbstractManifold, f, p_m; kwargs...)
    mco = ManifoldCostObjective(f)
    return cma_es!(M, mco, copy(M, p_m); kwargs...)
end

function cma_es!(M::AbstractManifold, f, p_m; kwargs...)
    mco = ManifoldCostObjective(f)
    return cma_es!(M, mco, p_m; kwargs...)
end

function default_cma_es_stopping_criterion(
    M::AbstractManifold, λ::Int; tol_fun::TParam=1e-12, tol_x::TParam=1e-12
) where {TParam<:Real}
    return StopAfterIteration(50000) |
           StopWhenCovarianceIllConditioned() |
           StopWhenBestCostInGenerationConstant{TParam}(
               Int(10 + ceil(30 * manifold_dimension(M) / λ))
           ) |
           StopWhenEvolutionStagnates(
               Int(120 + 30 * ceil(30 * manifold_dimension(M) / λ)), 20000, 0.3
           ) |
           StopWhenPopulationDiverges(1e4) |
           StopWhenPopulationCostConcentrated(
               tol_fun, Int(10 + ceil(30 * manifold_dimension(M) / λ))
           ) |
           StopWhenPopulationStronglyConcentrated(tol_x)
end

function cma_es!(
    M::AbstractManifold,
    mco::O,
    p_m;
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
    basis::AbstractBasis=default_basis(M, typeof(p_m)),
    rng::AbstractRNG=default_rng(),
    kwargs..., #collect rest
) where {O<:Union{AbstractManifoldCostObjective,AbstractDecoratedManifoldObjective}}
    dmco = decorate_objective!(M, mco; kwargs...)
    mp = DefaultManoptProblem(M, dmco)
    n_coords = number_of_coordinates(M, basis)
    wp = [log((λ + 1) / 2) - log(i) for i in 1:λ] # Eq. (49)
    μ = Int(floor(λ / 2)) # Table 1 caption
    μ_eff = (sum(wp[1:μ])^2) / (sum(x -> x^2, wp[1:μ]))  # Table 1 caption
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

    c_σ = (μ_eff + 2) / (n_coords + μ_eff + 5) # Eq. (55)
    d_σ = 1 + 2 * max(0, sqrt((μ_eff - 1) / (n_coords + 1)) - 1) + c_σ # Eq. (55)
    c_c = (4 + μ_eff / n_coords) / (n_coords + 4 + 2 * μ_eff / n_coords) # Eq. (56)
    covariance_matrix = Matrix{number_eltype(p_m)}(I, n_coords, n_coords)
    state = CMAESState(
        M,
        p_m,
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
        covariance_matrix,
        σ,
        recombination_weights;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
        basis=basis,
        rng=rng,
    )

    d_state = decorate_state!(state; kwargs...)
    solve!(mp, d_state)
    return get_solver_return(get_objective(mp), d_state)
end

@doc raw"""
    eigenvector_transport!(
        M::AbstractManifold,
        matrix_eigen::Eigen,
        p,
        q,
        basis::AbstractBasis,
        vtm::AbstractVectorTransportMethod,
    )

Transport the matrix with `matrix_eig` eigen decomposition when expanded in `basis` from
point `p` to point `q` on `M`. Update `matrix_eigen` in-place.

`(p, matrix_eig)` belongs to the fiber bundle of ``B = \mathcal M × SPD(n)``, where `n`
is the (real) dimension of `M`. The function corresponds to the Ehresmann connection
defined by vector transport `vtm` of eigenvectors of `matrix_eigen`.
"""
function eigenvector_transport!(
    M::AbstractManifold,
    matrix_eigen::Eigen,
    p,
    q,
    basis::AbstractBasis,
    vtm::AbstractVectorTransportMethod,
)
    if is_flat(M)
        return matrix_eigen
    end
    n = length(matrix_eigen.values)
    X = zero_vector(M, p)
    for i in 1:n
        get_vector!(M, X, p, view(matrix_eigen.vectors, :, i), basis)
        vector_transport_to!(M, X, p, X, q, vtm)
        get_coordinates!(M, view(matrix_eigen.vectors, :, i), q, X, basis)
    end
    return matrix_eigen
end

"""
    StopWhenCovarianceIllConditioned <: StoppingCriterion

Stop CMA-ES if condition number of covariance matrix exceeds `threshold`. This corresponds
to `ConditionCov` condition from [Hansen:2023](@cite).
"""
mutable struct StopWhenCovarianceIllConditioned{T<:Real} <: StoppingCriterion
    threshold::T
    last_cond::T
    at_iteration::Int
end
function StopWhenCovarianceIllConditioned(threshold::Real=1e14)
    return StopWhenCovarianceIllConditioned{typeof(threshold)}(threshold, 1, -1)
end

indicates_convergence(c::StopWhenCovarianceIllConditioned) = false
is_active_stopping_criterion(c::StopWhenCovarianceIllConditioned) = c.at_iteration > 0
function (c::StopWhenCovarianceIllConditioned)(
    ::AbstractManoptProblem, s::CMAESState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
        return false
    end
    c.last_cond = s.covariance_matrix_cond
    if k > 0 && c.last_cond > c.threshold
        c.at_iteration = k
        return true
    end
    return false
end
function status_summary(c::StopWhenCovarianceIllConditioned)
    has_stopped = c.at_iteration > 0
    s = has_stopped ? "reached" : "not reached"
    return "cond(s.covariance_matrix) > $(c.threshold):\t$s"
end
function get_reason(c::StopWhenCovarianceIllConditioned)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration) the condition number of covariance matrix ($(c.last_cond)) exceeded the threshold ($(c.threshold)).\n"
    end
    return ""
end
function show(io::IO, c::StopWhenCovarianceIllConditioned)
    return print(
        io, "StopWhenCovarianceIllConditioned($(c.threshold))\n    $(status_summary(c))"
    )
end

"""
    StopWhenBestCostInGenerationConstant <: StoppingCriterion

Stop if the range of the best objective function values of the last `iteration_range`
generations is zero. This corresponds to `EqualFUnValues` condition from
[Hansen:2023](@cite).

See also `StopWhenPopulationCostConcentrated`.
"""
mutable struct StopWhenBestCostInGenerationConstant{TParam<:Real} <: StoppingCriterion
    iteration_range::Int
    best_objective_at_last_change::TParam
    iterations_since_change::Int
    at_iteration::Int
end

function StopWhenBestCostInGenerationConstant{TParam}(iteration_range::Int) where {TParam}
    return StopWhenBestCostInGenerationConstant{TParam}(iteration_range, Inf, 0, -1)
end

# It just indicates stagnation, not that convergence to a minimizer
indicates_convergence(c::StopWhenBestCostInGenerationConstant) = true
function is_active_stopping_criterion(c::StopWhenBestCostInGenerationConstant)
    return c.iterations_since_change >= c.iteration_range
end
function (c::StopWhenBestCostInGenerationConstant)(
    ::AbstractManoptProblem, s::CMAESState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
        c.best_objective_at_last_change = Inf
        return false
    end
    if c.iterations_since_change >= c.iteration_range
        c.at_iteration = k
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
function status_summary(c::StopWhenBestCostInGenerationConstant)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "c.iterations_since_change > $(c.iteration_range):\t$s"
end
function get_reason(c::StopWhenBestCostInGenerationConstant)
    if c.at_iteration >= 0
        return "At iteration $(c.at_iteration): for the last $(c.iterations_since_change) generations the best objective value in each generation was equal to $(c.best_objective_at_last_change).\n"
    end
    return ""
end
function show(io::IO, c::StopWhenBestCostInGenerationConstant)
    return print(
        io,
        "StopWhenBestCostInGenerationConstant($(c.iteration_range))\n    $(status_summary(c))",
    )
end

"""
    StopWhenEvolutionStagnates{TParam<:Real} <: StoppingCriterion

The best and median fitness in each iteration is tracked over the last 20% but
at least `min_size` and no more than `max_size` iterations. Solver is stopped if
in both histories the median of the most recent `fraction` of values is not better
than the median of the oldest `fraction`.
"""
mutable struct StopWhenEvolutionStagnates{TParam<:Real} <: StoppingCriterion
    min_size::Int
    max_size::Int
    fraction::TParam
    best_history::CircularBuffer{TParam}
    median_history::CircularBuffer{TParam}
    at_iteration::Int
end

function StopWhenEvolutionStagnates(
    min_size::Int, max_size::Int, fraction::TParam
) where {TParam<:Real}
    return StopWhenEvolutionStagnates{TParam}(
        min_size,
        max_size,
        fraction,
        CircularBuffer{TParam}(max_size),
        CircularBuffer{TParam}(max_size),
        -1,
    )
end

# It just indicates stagnation, not convergence to a minimizer
indicates_convergence(c::StopWhenEvolutionStagnates) = true
function is_active_stopping_criterion(c::StopWhenEvolutionStagnates)
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
function (c::StopWhenEvolutionStagnates)(::AbstractManoptProblem, s::CMAESState, k::Int)
    if k == 0 # reset on init
        empty!(c.best_history)
        empty!(c.median_history)
        c.at_iteration = -1
        return false
    end
    if is_active_stopping_criterion(c)
        c.at_iteration = k
        return true
    else
        push!(c.best_history, s.best_fitness_current_gen)
        push!(c.median_history, s.median_fitness_current_gen)
    end
    return false
end
function status_summary(c::StopWhenEvolutionStagnates)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    N = length(c.best_history)
    if N == 0
        return "best and median fitness not yet filled, stopping criterion:\t$s"
    end
    thr_low = Int(ceil(N * c.fraction))
    thr_high = Int(floor(N * (1 - c.fraction)))
    median_best_old = median(c.best_history[1:thr_low])
    median_best_new = median(c.best_history[thr_high:end])
    median_median_old = median(c.median_history[1:thr_low])
    median_median_new = median(c.median_history[thr_high:end])
    return "generation >= $(c.min_size) && $(median_best_old) <= $(median_best_new) && $(median_median_old) <= $(median_median_new):\t$s"
end
function get_reason(c::StopWhenEvolutionStagnates)
    if c.at_iteration >= 0
        return "Both median and best objective history became stagnant.\n"
    end
    return ""
end
function show(io::IO, c::StopWhenEvolutionStagnates)
    return print(
        io,
        "StopWhenEvolutionStagnates($(c.min_size), $(c.max_size), $(c.fraction))\n    $(status_summary(c))",
    )
end

@doc raw"""
    StopWhenPopulationStronglyConcentrated{TParam<:Real} <: StoppingCriterion

Stop if the standard deviation in all coordinates is smaller than `tol` and
norm of `σ * p_c` is smaller than `tol`. This corresponds to `TolX` condition from
[Hansen:2023](@cite).

# Fields

* `tol` the tolerance to verify against
* `at_iteration` an internal field to indicate at with iteration ``i \geq 0`` the tolerance was met.

# Constructor

    StopWhenPopulationStronglyConcentrated(tol::Real)
"""
mutable struct StopWhenPopulationStronglyConcentrated{TParam<:Real} <: StoppingCriterion
    tol::TParam
    at_iteration::Int
end
function StopWhenPopulationStronglyConcentrated(tol::Real)
    return StopWhenPopulationStronglyConcentrated{typeof(tol)}(tol, -1)
end

# It just indicates stagnation, not convergence to a minimizer
indicates_convergence(c::StopWhenPopulationStronglyConcentrated) = true
function is_active_stopping_criterion(c::StopWhenPopulationStronglyConcentrated)
    return c.at_iteration >= 0
end
function (c::StopWhenPopulationStronglyConcentrated)(
    ::AbstractManoptProblem, s::CMAESState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
        return false
    end
    norm_inf_dev = norm(s.deviations, Inf)
    norm_inf_p_c = norm(s.p_c, Inf)
    if norm_inf_dev < c.tol && s.σ * norm_inf_p_c < c.tol
        c.at_iteration = k
        return true
    end
    return false
end
function status_summary(c::StopWhenPopulationStronglyConcentrated)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "norm(s.deviations, Inf) < $(c.tol) && norm(s.σ * s.p_c, Inf) < $(c.tol) :\t$s"
end
function get_reason(c::StopWhenPopulationStronglyConcentrated)
    if c.at_iteration >= 0
        return "Standard deviation in all coordinates is smaller than $(c.tol) and `σ * p_c` has Inf norm lower than $(c.tol).\n"
    end
    return ""
end
function show(io::IO, c::StopWhenPopulationStronglyConcentrated)
    return print(
        io, "StopWhenPopulationStronglyConcentrated($(c.tol))\n    $(status_summary(c))"
    )
end

"""
    StopWhenPopulationDiverges{TParam<:Real} <: StoppingCriterion

Stop if `σ` times maximum deviation increased by more than `tol`. This usually indicates a
far too small `σ`, or divergent behavior. This corresponds to `TolXUp` condition from
[Hansen:2023](@cite).
"""
mutable struct StopWhenPopulationDiverges{TParam<:Real} <: StoppingCriterion
    tol::TParam
    last_σ_times_maxstddev::TParam
    at_iteration::Int
end
function StopWhenPopulationDiverges(tol::Real)
    return StopWhenPopulationDiverges{typeof(tol)}(tol, 1.0, -1)
end

indicates_convergence(c::StopWhenPopulationDiverges) = false
function is_active_stopping_criterion(c::StopWhenPopulationDiverges)
    return c.at_iteration >= 0
end
function (c::StopWhenPopulationDiverges)(::AbstractManoptProblem, s::CMAESState, k::Int)
    if k == 0 # reset on init
        c.at_iteration = -1
        return false
    end
    cur_σ_times_maxstddev = s.σ * maximum(s.deviations)
    if cur_σ_times_maxstddev / c.last_σ_times_maxstddev > c.tol
        c.at_iteration = k
        return true
    end
    return false
end
function status_summary(c::StopWhenPopulationDiverges)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "cur_σ_times_maxstddev / c.last_σ_times_maxstddev > $(c.tol) :\t$s"
end
function get_reason(c::StopWhenPopulationDiverges)
    if c.at_iteration >= 0
        return "σ times maximum standard deviation exceeded $(c.tol). This indicates either much too small σ or divergent behavior.\n"
    end
    return ""
end
function show(io::IO, c::StopWhenPopulationDiverges)
    return print(io, "StopWhenPopulationDiverges($(c.tol))\n    $(status_summary(c))")
end

"""
    StopWhenPopulationCostConcentrated{TParam<:Real} <: StoppingCriterion

Stop if the range of the best objective function value in the last `max_size` generations
and all function values in the current generation is below `tol`. This corresponds to
`TolFun` condition from [Hansen:2023](@cite).

# Constructor

    StopWhenPopulationCostConcentrated(tol::Real, max_size::Int)
"""
mutable struct StopWhenPopulationCostConcentrated{TParam<:Real} <: StoppingCriterion
    tol::TParam
    best_value_history::CircularBuffer{TParam}
    at_iteration::Int
end
function StopWhenPopulationCostConcentrated(tol::TParam, max_size::Int) where {TParam<:Real}
    return StopWhenPopulationCostConcentrated{TParam}(
        tol, CircularBuffer{TParam}(max_size), -1
    )
end

# It just indicates stagnation, not convergence to a minimizer
indicates_convergence(c::StopWhenPopulationCostConcentrated) = true
function is_active_stopping_criterion(c::StopWhenPopulationCostConcentrated)
    return c.at_iteration >= 0
end
function (c::StopWhenPopulationCostConcentrated)(
    ::AbstractManoptProblem, s::CMAESState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
        return false
    end
    push!(c.best_value_history, s.best_fitness_current_gen)
    if isfull(c.best_value_history)
        min_hist, max_hist = extrema(c.best_value_history)
        if max_hist - min_hist < c.tol &&
            s.best_fitness_current_gen - s.worst_fitness_current_gen < c.tol
            c.at_iteration = k
            return true
        end
    end
    return false
end
function status_summary(c::StopWhenPopulationCostConcentrated)
    has_stopped = is_active_stopping_criterion(c)
    s = has_stopped ? "reached" : "not reached"
    return "range of best objective values in the last $(length(c.best_value_history)) generations and all objective values in the current one < $(c.tol) :\t$s"
end
function get_reason(c::StopWhenPopulationCostConcentrated)
    if c.at_iteration >= 0
        return "Range of best objective function values in the last $(length(c.best_value_history)) generations and all values in the current generation is below $(c.tol)\n"
    end
    return ""
end
function show(io::IO, c::StopWhenPopulationCostConcentrated)
    return print(
        io, "StopWhenPopulationCostConcentrated($(c.tol))\n    $(status_summary(c))"
    )
end
