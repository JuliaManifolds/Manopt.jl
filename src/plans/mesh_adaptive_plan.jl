"""
    AbstractMeshPollFunction

An abstract type for common “poll” strategies in the [`mesh_adaptive_direct_search`](@ref)
solver.
A subtype of this The functor has to fulllfil

* be callable as `poll!(problem, mesh_size; kwargs...)` and modify the state

as well as

* provide a `get_poll_success(poll!)` function that indicates whether the last poll was successful in finding a new candidate,
this returns the last successful mesh vector used.

The `kwargs...` could include
* `scale_mesh=1.0`: to rescale the mesh globally
* `max_stepsize=Inf`: avoid exceeding a step size beyon this, e.g. injectivity radius.
  any vector longer than this should be shortened to the provided max stepsize.
"""
abstract type AbstractMeshPollFunction end

"""
    AbstractMeshSearchFunction

Should be callable as search!(problem, mesh_size, p, X; kwargs...)

where `X` is the last succesful poll direction from the tangent space at `p``
if that exists and the zero vector otherwise.
"""
abstract type AbstractMeshSearchFunction end

"""
    StopWhenPollSizeLess <: StoppingCriterion

stores a threshold when to stop looking at the poll mesh size of an [`MeshAdaptiveDirectSearchState`](@ref).

# Constructor

    StopWhenPollSizeLess(ε)

initialize the stopping criterion to a threshold `ε`.
"""
mutable struct StopWhenPollSizeLess{F} <: StoppingCriterion
    threshold::F
    last_poll_size::F
    at_iteration::Int
    function StopWhenPollSizeLess(ε::F) where {F<:Real}
        return new{F}(ε, zero(ε), -1)
    end
end

#
# Specific Polls
#
"""
    LowerTriangularAdaptivePoll <: AbstractMeshPollFunction

Generate a mesh (poll step) based on Section 6 and 7 of [Deisigmeyer:2007](@ref),
with two small modifications:
* The mesh can be scaled globally so instead of ``Δ_0^m=1`` a certain different scale is used
* Any poll direction can be rescaled if it is too long. This is to not exceed the inhectivity radius for example.

# Functor

    (p::LowerTriangularAdaptivePoll)(problem, mesh_size; scale_mesh=1.0, max_stepsize=inf)



# Fields

* `p::P`: a point on the manifold, where the mesh is build in the tangent space
* `q::P`: a memory for a new point/candidate
* `random_vector`: a ``d``-dimensional random vector ``b_l```
* `random_index`: a random index ``ι``
* `mesh`: a vector of tangent vectors storing the mesh.
* `basis`: a basis of the current tangent space with respect to which the mesh is stored
* `X::T` the last successful poll direction stored as a tangent vector.
  initialised to the zero vector and reset to the zero vector after moving to a new tangent space.
$(_var(:Field, :retraction_method))
$(_var(:Field, :vector_transport_method))
"""
mutable struct LowerTriangularAdaptivePoll{
    P,
    T,
    F<:Real,
    V<:AbstractVector{F},
    M<:AbstractMatrix{F},
    I<:Int,
    B,
    VTM<:AbstractVectorTransportMethod,
    RM<:AbstractRetractionMethod,
} <: AbstractMeshPollFunction
    p::P
    q::P
    random_vector::V
    random_index::I
    mesh::M
    basis::B
    X::T
    last_poll_improved::Bool
    retraction_method::RM
    vector_transport_method::VTM
end

function LowerTriangularAdaptivePoll(
    M,
    p=rand(M);
    basis=DefaultOrthonormalBasis(),
    retraction_method=default_retraction_method(M),
    vector_transport_method=default_vector_transport_method(M),
)
    d = manifold_dimension(M)
    b_l = zeros(d)
    D_k = zeros(d, d + 1)
    X = zero_vector(M, p)
    return LowerTriangularAdaptiveMesh(
        p,
        copy(M, p),
        0,
        b_l,
        0,
        D_k,
        basis,
        X,
        false,
        retraction_method,
        vector_transport_method,
    )
end
function get_poll_success(ltap::LowerTriangularAdaptivePoll)
    return ltap.last_poll_improved
end
function get_poll_direction(ltap::LowerTriangularAdaptivePoll)
    return ltap.X
end
function get_poll_basepoint(ltap::LowerTriangularAdaptivePoll)
    return ltap.p
end
function update_poll_basepoint!(M, ltap::LowerTriangularAdaptivePoll{P}, p::P) where {P}
    vector_transport_to!(M, ltap.X, ltap.p, ltap.X, p, ltap.vector_transport_method)
    return copyto!(M, ltap.p, p)
end
function show(io::IO, ltap::LowerTriangularAdaptivePoll)
    s = "LowerTriangularAdaptivePoll using `basis=`$(ltap.basis), `retraction_method=`$(ltap.retraction_method), and `vector_transport_method=`$(ltap.vector_transport_method)"
    return print(io, s)
end
function (ltap::LowerTriangularAdaptivePoll)(
    amp::AbstractManoptProblem, mesh_size; scale_mesh=1.0, max_stepsize=inf
)
    M = get_manifold(amp)
    n = manifold_dimension(M)
    l = -log(4, mesh_size)
    S = (-2^l + 1):(2^l - 1)
    # A random index ι
    ltap.random_index = rand(1:n)
    # generate a random b_l vector
    for i in 1:n
        ltap.random_vector[i] = rand(i == ltap.random_index ? [-2^l, 2^l] : S)
    end
    # Generate L lower triangular, (n-1)x(n-1) in M
    for i in 1:(n - 1)
        for j in (n - 1)
            ltap.mesh[i, j] = (j > i) ? 0.0 : rand((i == j) ? [-2^l, 2^l] : S)
        end
    end
    # Shift to construct n × n matrix B
    # (bottom left)
    ltap.mesh[(ltap.random_index + 1):n, (1:n)] = poll.mesh[
        (ltap.random_index):(n - 1), (1:n)
    ]
    # zero row above
    ltap.mesh[ltap.random_index, (1:n)] .= 0
    # left column: random vector
    ltap.mesh[:, n] .= ltap.random_vector
    # set last column to minus the sum.
    ltap.mesh[:, n + 1] .= -1 .* sum(ltap.mesh[:, 1:n]; dims=2)
    # Permute B (first n columns)
    ltap.mesh[:, 1:n] .= ltap.mesh[:, randperm(n)]
    # look for best
    ltap.last_poll_improved = false
    i = 0
    c = get_cost(amp, ltap.p)
    while (i < (n + 1)) && !(ltap.last_poll_improved)
        i = i + 1 # runs for the last time for i=n+1 and hence the sum.
        # get vector – scale mesh
        get_vector!(M, ltap.X, ltap.p, scale_mesh .* ltap.mesh[:, i], ltap.basis)
        # shorten if necessary
        if norm(M, ltap, ltap.p, ltap.X) > max_stepsize
            ltap.X = max_stepsize & norm(M, ltap, ltap.p, ltap.X) * ltap.X
        end
        retract!(M, ltap.q, ltap.p, ltap.X, ltap.retraction_method)
        if get_cost(amp, ltap.q) < c
            copyto!(M, ltap.p, ltap.q)
            ltap.last_poll_improved = true
            # this also breaks while
        end
    end
    # clear temp vector if we did not improve.
    !(ltap.last_poll_improved) && (zero_vector!(M, ltap.X, p))
    return ltap
end

#
# Specific Searches
#

"""
    DefaultMeshAdaptiveDirectSearch <: AbstractMeshSearchFunction

# Functor

    (s::DefaultMeshAdaptiveDirectSearch)(problem, mesh_size, X; scale_mesh=1.0, max_stepsize=inf)

# Fields

* `q`: a temporary memory for a point on the manifold
* `X`: information to perform the search, e.g. the last direction found by poll.
* `last_seach_improved::Bool` indicate whether the last search was succesfull, i.e. improved the cost.
$(_var(:Field, :retraction_method))

# Constructor

    DefaultMeshAdaptiveDirectSearch(M::AbstractManifold, p=rand(M); kwargs...)

## Keyword awrguments

* `X::T=zero_vector(M,p)
$(_var(:Keyword, :retraction_method))
"""
mutable struct DefaultMeshAdaptiveDirectSearch{P,T,RM} <: AbstractMeshSearchFunction
    p::P
    X::T
    last_search_improved::Bool
    retraction_method::RM
end
function DefaultMeshAdaptiveDirectSearch(
    M, p=rand(M); X=zero_vector(M, p), retraction_method=default_retaction_method(M)
)
    return DefaultMeshAdaptiveDirectSearch(p, X, false, retraction_method)
end
function get_search_success(search!::DefaultMeshAdaptiveDirectSearch)
    return search!.last_search_improved
end
function get_search_point(search!::DefaultMeshAdaptiveDirectSearch)
    return search!.last_search_improved
end
function show(io::IO, dmads::DefaultMeshAdaptiveDirectSearch)
    s = "DefaultMeshAdaptiveDirectSearch using `retraction_method=`$(dmads.retraction_method)"
    return print(io, s)
end
function (dmads::DefaultMeshAdaptiveDirectSearch)(
    amp::AbstractManoptProblem, mesh_size, p, X; scale_mesh=1.0, max_stepsize=inf
)
    M = get_manifold(amp)
    dmads.X = 4 * mesh_size * scale_mesh * X
    if norm(M, p, dmads.X) > max_stepsize
        dmads.X = max_stepsize / norm(M, dmads.p, dmads.X) * dmads.X
    end
    retract!(M, dmads.p, p, dmads.X, dmads.retraction_method)
    dmads.last_search_improved = get_cost(amp, dmads.q) < get_cost(amp, p)
    # Implement the code from Dreisigmeyer p. 17 about search generation
    return dmads
end

"""
    MeshAdaptiveDirectSearchState <: AbstractManoptSolverState

* `p`: current iterate
* `q`: temp (old) iterate

"""
mutable struct MeshAdaptiveDirectSearchState{P,F<:Real,M,PT,ST,TStop<:StoppingCriterion} <:
               AbstractManoptSolverState
    p::P
    q::P
    mesh_size::F
    scale_mesh::F
    max_stepsize::F
    poll_size::F
    stop::TStop
    poll::PT
    search::ST
end
function MeshAdaptiveDirectSearchState(
    M::AbstractManifold,
    p=rand(M);
    mesh_basis::B=DefaultOrthonormalBasis(),
    mesh_size=injectivity_radius(M) / 4,
    scale_mesh=1.0,
    max_stepsize=injectivity_radius(M),
    poll_size=manifold_dimension(M) * sqrt(mesh_size),
    stopping_criterion::SC=StopAfterIteration(500) | StopWhenPollSizeLess(1e-7),
    retraction_method=default_retraction_method(M, typeof(p)),
    vector_transport_method=default_vector_transport_method(M, typeof(p)),
    poll::PT=LowerTriangularAdaptivePoll(
        M,
        copy(M, p);
        basis=mesh_basis,
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
    ),
    search::ST=DefaultMeshAdaptiveDirectSearch(
        M, copy(M, p); retraction_method=retraction_method
    ),
) where {
    PT<:AbstractMeshPollFunction,
    ST<:AbstractMeshSearchFunction,
    SC<:StoppingCriterion,
    B<:AbstractBasis,
}
    return MeshAdaptiveDirectSearchState(
        p,
        copy(p),
        mesh_size,
        scale_mesh,
        max_stepsize,
        poll_size,
        stopping_criterion,
        poll,
        search,
    )
end

function show(io::IO, mads::MeshAdaptiveDirectSearchState)
    i = get_count(mads, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    s = """
    # Solver state for `Manopt.jl`s mesh adaptive direct search
    $Iter

    ## Parameters
    * `mesh_size = ` $(mads.mesh_size)
    * `scale_mesh = ` $(mads.scale_mesh)
    * `max_stepsize = ` $(mads.max_stepsize)
    * `poll_size = ` $(mads.poll_size)
    * `poll:` $(repr(mads.poll))`
    * `search:` $(repr(mads.poll))`

    ## Stopping criterion
    $(status_summary(mads.stop))

    This indicates convergence: $Conv"""
    return print(io, s)
end

get_solver_result(ips::MeshAdaptiveDirectSearchState) = ips.p

function (c::StopWhenPollSizeLess)(
    p::AbstractManoptProblem, s::MeshAdaptiveDirectSearchState, k::Int
)
    if k == 0 # reset on init
        c.at_iteration = -1
    end
    c.last_poll_size = s.poll_size
    if c.last_poll_size < c.threshold && k > 0
        c.at_iteration = k
        return true
    end
    return false
end
function get_reason(c::StopWhenPollSizeLess)
    if (c.last_poll_size < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm computed a poll step size ($(c.last_poll_size)) less than $(c.threshold).\n"
    end
    return ""
end
function status_summary(c::StopWhenPollSizeLess)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return "Poll step size s < $(c.threshold):\t$s"
end
function show(io::IO, c::StopWhenPollSizeLess)
    return print(io, "StopWhenPollSizeLess($(c.threshold))\n    $(status_summary(c))")
end
