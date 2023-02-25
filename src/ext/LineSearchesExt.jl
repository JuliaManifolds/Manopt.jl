
"""
    LineSearchesStepsize <: Stepsize

Wrapper for line searches available in the `LineSearches.jl` library.

## Constructors

    LineSearchesStepsize(
        M::AbstractManifold,
        linesearch,
        initial_stepsize;
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(M),
    )
    LineSearchesStepsize(
        linesearch,
        initial_stepsize;
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    )

Wrap `linesearch` (for example [`HagerZhang`](https://julianlsolvers.github.io/LineSearches.jl/latest/reference/linesearch.html#LineSearches.HagerZhang)
or [`MoreThuente`)(https://julianlsolvers.github.io/LineSearches.jl/latest/reference/linesearch.html#LineSearches.MoreThuente))
and initial step selector `initial_stepsize` that will work on manifold `M`. Retraction used
for determining the line along which the search initial_stepsize performed can be provided
as `retraction_method`. Gradient vectors are transported between points using
`vector_transport_method`.
"""
struct LineSearchesStepsize{
    TLS,TIS,TRM<:AbstractRetractionMethod,TVTM<:AbstractVectorTransportMethod
} <: Stepsize
    linesearch::TLS
    initial_step::TIS
    retraction_method::TRM
    vector_transport_method::TVTM
end
function LineSearchesStepsize(
    M::AbstractManifold,
    linesearch,
    initial_stepsize;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M
    ),
)
    return LineSearchesStepsize(
        linesearch,
        initial_stepsize;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    )
end
function LineSearchesStepsize(
    linesearch,
    initial_stepsize;
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
)
    return LineSearchesStepsize{
        typeof(linesearch),
        typeof(initial_stepsize),
        typeof(retraction_method),
        typeof(vector_transport_method),
    }(
        linesearch, initial_stepsize, retraction_method, vector_transport_method
    )
end

stepsize_storage_points(::LineSearchesStepsize) = Tuple{:p_tmp}

stepsize_storage_vectors(::LineSearchesStepsize) = Tuple{:X_tmp,:Y_tmp}

function (cs::StepsizeStorage{<:LineSearchesStepsize})(
    mp::AbstractManoptProblem,
    s::AbstractManoptSolverState,
    i::Int,
    η=-get_gradient(s);
    fp=get_cost(mp, get_iterate(s)),
    kwargs...,
)
    css = cs.stepsize
    M = get_manifold(mp)
    p = get_iterate(s)
    X = get_gradient(s)
    p_tmp = get_storage(cs.storage, PointStorageKey(:p_tmp))
    X_tmp = get_storage(cs.storage, VectorStorageKey(:X_tmp))
    Y_tmp = get_storage(cs.storage, VectorStorageKey(:Y_tmp))
    f = get_cost_function(get_objective(mp))
    dphi_0 = real(inner(M, p, X, η))

    # guess initial alpha
    get_initial_alpha(M, css.initial_step, cs, p, η, fp, dphi_0)
    α0 = cs.last_stepsize

    cs.initial_cost = fp

    # perform actual linesearch

    function ϕ(α)
        retract!(M, p_tmp, p, η, α, css.retraction_method)
        return f(M, p_tmp)
    end
    function dϕ(α)
        retract!(M, p_tmp, p, η, α, css.retraction_method)
        get_gradient!(mp, X_tmp, p_tmp)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, css.vector_transport_method)
        return real(inner(M, p_tmp, Y_tmp, Y_tmp))
    end
    function ϕdϕ(α)
        # TODO: optimize?
        retract!(M, p_tmp, p, η, α, css.retraction_method)
        get_gradient!(mp, X_tmp, p_tmp)
        vector_transport_to!(M, Y_tmp, p, η, p_tmp, css.vector_transport_method)
        phi = f(M, p_tmp)
        dphi = real(inner(M, p_tmp, Y_tmp, Y_tmp))
        return (phi, dphi)
    end

    try
        α, fp = css.linesearch(ϕ, dϕ, ϕdϕ, α0, fp, dphi_0)
        return α
    catch ex
        if isa(ex, LineSearches.LineSearchException)
            # maybe indicate failure?
            return zero(dphi_0)
        else
            rethrow(ex)
        end
    end
end

function show(io::IO, cs::LineSearchesStepsize)
    return print(
        io,
        "LineSearchesStepsize($(cs.linesearch); retraction_method=$(cs.retraction_method), vector_transport_method=$(cs.vector_transport_method))",
    )
end

"""
    nanmax(x, y)

Maximum of arguments, ignoring those that are NaN.
"""
function nanmax(x, y)
    isnan(x) && return y
    isnan(y) && return x
    return max(x, y)
end
function nanmax(x, y, z)
    return nanmax(nanmax(x, y), z)
end

"""
    nanmin(x, y)

Minimum of arguments, ignoring those that are NaN.
"""
function nanmin(x, y)
    isnan(x) && return y
    isnan(y) && return x
    return min(x, y)
end

# code below was adapted from https://github.com/JuliaNLSolvers/LineSearches.jl/blob/master/src/initialguess.jl

"""
    get_initial_alpha(
        M::AbstractManifold, initial_stepsize, state::StepsizeStorage, p, η, phi_0, dphi_0
    )

Calculate initial stepsize using method `initial_stepsize` from LineSearches.jl, when
optimizing on manifold `M` performing line search starting at point `p` in direction `η`,
when line search state is equal `state`, objective value at `p` is equal to `phi_0` and
real component of inner product between gradient and line search direction at `p` is
equal to `dphi_0`.
"""
get_initial_alpha(
    M::AbstractManifold, initial_stepsize, state::StepsizeStorage, p, η, phi_0, dphi_0
)

function get_initial_alpha(
    M::AbstractManifold,
    initial_stepsize::InitialStatic{T},
    state::StepsizeStorage,
    p,
    η,
    phi_0,
    dphi_0,
) where {T}
    PT = promote_type(T, real(number_eltype(η)))
    if initial_stepsize.scaled == true && (ns = real(norm(M, p, η))) > convert(PT, 0)
        state.last_stepsize = convert(PT, min(initial_stepsize.alpha, ns)) / ns
    else
        state.last_stepsize = convert(PT, initial_stepsize.alpha)
    end
end

function get_initial_alpha(
    ::AbstractManifold,
    initial_stepsize::InitialPrevious,
    state::StepsizeStorage,
    p,
    η,
    phi_0,
    dphi_0,
)
    if isnan(state.last_stepsize)
        state.last_stepsize = initial_stepsize.alpha
    end
    state.last_stepsize = min(initial_stepsize.alphamax, state.last_stepsize)
    return state.last_stepsize = max(initial_stepsize.alphamin, state.last_stepsize)
end

function get_initial_alpha(
    ::AbstractManifold,
    initial_stepsize::InitialQuadratic{T},
    state::StepsizeStorage,
    p,
    η,
    phi_0,
    dphi_0,
) where {T}
    if !isfinite(state.initial_cost) || isapprox(dphi_0, convert(T, 0); atol=eps(T))
        # If we're at the first iteration
        αguess = initial_stepsize.α0
    else
        αguess = 2 * (phi_0 - state.initial_cost) / dphi_0
        αguess = nanmax(
            initial_stepsize.αmin, state.last_stepsize * initial_stepsize.ρ, αguess
        )
        αguess = nanmin(initial_stepsize.αmax, αguess)
        # if αguess ≈ 1, then make it 1 (Newton-type behaviour)
        if initial_stepsize.snap2one[1] ≤ αguess ≤ initial_stepsize.snap2one[2]
            αguess = one(state.last_stepsize)
        end
    end
    return state.last_stepsize = αguess
end

function get_initial_alpha(
    ::AbstractManifold,
    initial_stepsize::InitialConstantChange{T},
    state::StepsizeStorage,
    p,
    η,
    phi_0,
    dphi_0,
) where {T}
    if !isfinite(initial_stepsize.dϕ_0_previous[]) ||
        !isfinite(state.last_stepsize) ||
        isapprox(dphi_0, convert(T, 0); atol=eps(T))
        # If we're at the first iteration
        αguess = initial_stepsize.α0
    else
        # state.last_stepsize initial_stepsize the previously used step length
        αguess = state.last_stepsize * initial_stepsize.dϕ_0_previous[] / dphi_0
        αguess = nanmax(
            initial_stepsize.αmin, state.last_stepsize * initial_stepsize.ρ, αguess
        )
        αguess = nanmin(initial_stepsize.αmax, αguess)
        # if αguess ≈ 1, then make it 1 (Newton-type behaviour)
        if initial_stepsize.snap2one[1] ≤ αguess ≤ initial_stepsize.snap2one[2]
            αguess = one(state.last_stepsize)
        end
    end
    initial_stepsize.dϕ_0_previous[] = dphi_0
    return state.last_stepsize = αguess
end

function get_last_stepsize(
    ::AbstractManoptProblem,
    ::AbstractManoptSolverState,
    step::StepsizeStorage{<:LineSearchesStepsize},
    args...,
)
    return step.last_stepsize
end
