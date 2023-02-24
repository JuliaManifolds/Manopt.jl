
"""
    LineSearchesStepsize <: Stepsize

Wrapper for line searches available in the `LineSearches.jl` library.

## Constructors

    LineSearchesStepsize(
        M::AbstractManifold,
        ls,
        is;
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(M),
    )
    LineSearchesStepsize(
        ls,
        is;
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(M),
    )

Wrap linesearch `ls` (for example `HagerZhang` or `MoreThuente`) and initial step selector
`is` that will work on manifold `M`. Retraction used for determining the line along which
the search is performed can be provided as `retraction_method`. Gradient vectors are
transported between points using `vector_transport_method`.
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
    ls,
    is;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M
    ),
)
    return LineSearchesStepsize(
        ls,
        is;
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    )
end
function LineSearchesStepsize(
    ls,
    is;
    retraction_method=ExponentialRetraction(),
    vector_transport_method=ParallelTransport(),
)
    return LineSearchesStepsize{
        typeof(ls),typeof(is),typeof(retraction_method),typeof(vector_transport_method)
    }(
        ls, is, retraction_method, vector_transport_method
    )
end

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
    p_tmp = copy(M, p)
    X_tmp = copy(M, p, η)
    Y_tmp = copy(M, p, η)
    f = get_cost_function(get_objective(mp))
    dphi_0 = real(inner(M, p, X, η))
    #println("dphi_0 = ", dphi_0)

    # guess initial alpha
    get_initial_alpha(M, css.initial_step, cs, p, η, fp, dphi_0)
    α0 = cs.alpha

    cs.last_f_p = fp

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

function get_initial_alpha(
    M::AbstractManifold, is::InitialStatic{T}, state::StepsizeStorage, p, η, phi_0, dphi_0
) where {T}
    PT = promote_type(T, real(number_eltype(η)))
    if is.scaled == true && (ns = real(norm(M, p, η))) > convert(PT, 0)
        # TODO: Type instability if there's a type mismatch between is.alpha and ns?
        state.alpha = convert(PT, min(is.alpha, ns)) / ns
    else
        state.alpha = convert(PT, is.alpha)
    end
end

function get_initial_alpha(
    ::AbstractManifold, is::InitialPrevious, state::StepsizeStorage, p, η, phi_0, dphi_0
)
    if isnan(state.alpha)
        state.alpha = is.alpha
    end
    state.alpha = min(is.alphamax, state.alpha)
    return state.alpha = max(is.alphamin, state.alpha)
end

function get_initial_alpha(
    ::AbstractManifold, is::InitialQuadratic{T}, state::StepsizeStorage, p, η, phi_0, dphi_0
) where {T}
    if !isfinite(state.last_f_p) || isapprox(dphi_0, convert(T, 0); atol=eps(T)) # Need to add a tolerance
        # If we're at the first iteration
        αguess = is.α0
    else
        αguess = 2 * (phi_0 - state.last_f_p) / dphi_0
        αguess = nanmax(is.αmin, state.alpha * is.ρ, αguess)
        αguess = nanmin(is.αmax, αguess)
        # if αguess ≈ 1, then make it 1 (Newton-type behaviour)
        if is.snap2one[1] ≤ αguess ≤ is.snap2one[2]
            αguess = one(state.alpha)
        end
    end
    return state.alpha = αguess
end

function get_initial_alpha(
    ::AbstractManifold,
    is::InitialConstantChange{T},
    state::StepsizeStorage,
    p,
    η,
    phi_0,
    dphi_0,
) where {T}
    if !isfinite(is.dϕ_0_previous[]) ||
        !isfinite(state.alpha) ||
        isapprox(dphi_0, convert(T, 0); atol=eps(T))
        # If we're at the first iteration
        αguess = is.α0
    else
        # state.alpha is the previously used step length
        αguess = state.alpha * is.dϕ_0_previous[] / dphi_0
        αguess = nanmax(is.αmin, state.alpha * is.ρ, αguess)
        αguess = nanmin(is.αmax, αguess)
        # if αguess ≈ 1, then make it 1 (Newton-type behaviour)
        if is.snap2one[1] ≤ αguess ≤ is.snap2one[2]
            αguess = one(state.alpha)
        end
    end
    is.dϕ_0_previous[] = dphi_0
    return state.alpha = αguess
end

function get_last_stepsize(
    ::AbstractManoptProblem,
    ::AbstractManoptSolverState,
    step::StepsizeStorage{<:LineSearchesStepsize},
    args...,
)
    return step.alpha
end
