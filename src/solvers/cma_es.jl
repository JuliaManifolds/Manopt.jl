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
    p_m::P # point around which we search for new candidates
    σ::TParams # step size
    p_σ::Vector{TParams} # coordinates of a vector in T_{p_m} M
    p_c::Vector{TParams} # coordinates of a vector in T_{p_m} M
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

    return s
end
function step_solver!(mp::AbstractManoptProblem, s::CMAESState, iteration::Int)
    M = get_manifold(mp)
    n_coords = number_of_coordinates(M, s.basis)

    # sampling and evaluation of new solutions

    D2, B = eigen(Symmetric(s.covm))
    D = sqrt.(D2)
    cov_invsqrt = B * Diagonal(inv.(D)) * B'
    Y_m = zero_vector(M, s.p_m)
    for i in 1:(s.λ)
        mul!(s.ys_c[i], B, D .* randn(s.rng, n_coords)) # Eqs. (38) and (39)
        get_vector!(M, Y_m, s.p_m, s.ys_c[i], s.basis) # Eqs. (38) and (39)
        retract!(M, s.population[i], s.p_m, Y_m, s.σ, s.retraction_method) # Eq. (40)
    end
    fitness_vals = map(p -> get_cost(mp, p), s.population)
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

@doc raw"""
    cma_es(M, f, p_m=rand(M); σ::Real=1.0, kwargs...)

Perform covariance matrix adaptation evolutionary strategy search for global gradient-free
randomized optimization. It is suitable for complicated non-convex functions. It can be
reasonably expected to find global minimum within 3σ distance from `p_m`.

Implementation is based on [Hansen:2023](@cite) with basic adaptations to the Riemannian
setting.

# Input

* `M`       a manifold ``\mathcal M``
* `f`       a cost function ``f: \mathcal M→ℝ`` to find a minimizer ``p^*`` for
* `p_m`     an initial point `p`
* `σ`       initial standard deviation
* `λ`       population size (can be increased for a more thorough search)

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
    stopping_criterion::StoppingCriterion=StopAfterIteration(500),
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
        p_m,
        σ,
        zeros(typeof(c_1), n_coords),
        zeros(typeof(c_1), n_coords),
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
