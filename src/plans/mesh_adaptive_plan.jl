"""
    AbstractMeshPollFunction

An abstract type for common “poll” strategies in the [`mesh_adaptive_direct_search`](@ref)
solver.
A subtype of this The functor has to fulfil

* be callable as `poll!(problem, mesh_size; kwargs...)` and modify the state

as well as to provide functions

* `is_successful(poll!)` that indicates whether the last poll was successful in finding a new candidate
* `get_basepoint(poll!)` that returns the base point at which the mesh is build
* `get_candidate(poll!)` that returns the last found candidate if the poll was successful.
  Otherwise the base point is returned
* `get_descent_direction(poll!)` the the vector that points from the base point to the candidate.
  If the last poll was not successful, the zero vector is returned
* `update_basepoint!(M, poll!, p)` that updates the base point to `p` and all necessary
  internal data to a new point to build a mesh at

The `kwargs...` could include

* `scale_mesh=1.0`: to rescale the mesh globally
* `max_stepsize=Inf`: avoid exceeding a step size beyond this value, e.g. injectivity radius.
  any vector longer than this should be shortened to the provided maximum step size.
"""
abstract type AbstractMeshPollFunction end

"""
    AbstractMeshSearchFunction

Should be callable as `search!(problem, mesh_size, p, X; kwargs...)`

where `X` is the last successful poll direction from the tangent space at `p`
if that exists and the zero vector otherwise.


Besides that the following functions should be implemented

* `is_successful(search!)` that indicates whether the last search was successful in finding a new candidate
* `get_candidate(search!)` that returns the last found candidate
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
    function StopWhenPollSizeLess(ε::F) where {F <: Real}
        return new{F}(ε, zero(ε), -1)
    end
end

#
# Specific Polls
#
"""
    LowerTriangularAdaptivePoll <: AbstractMeshPollFunction

Generate a mesh (poll step) based on Section 6 and 7 of [Dreisigmeyer:2007](@cite),
with two small modifications:
* The mesh can be scaled globally so instead of ``Δ_0^m=1`` a certain different scale is used
* Any poll direction can be rescaled if it is too long. This is to not exceed the injectivity radius for example.

# Functor

    (p::LowerTriangularAdaptivePoll)(problem, mesh_size; scale_mesh=1.0, max_stepsize=inf)

# Fields

* `base_point::P`: a point on the manifold, where the mesh is build in the tangent space
* `basis`: a basis of the current tangent space with respect to which the mesh is stored
* `candidate::P`: a memory for a new point/candidate
* `mesh`: a vector of tangent vectors storing the mesh.
* `random_vector`: a ``d``-dimensional random vector ``b_l```
* `random_index`: a random index ``ι``
$(_fields([:retraction_method, :vector_transport_method]))
* `X::T` the last successful poll direction stored as a tangent vector.
  initialised to the zero vector and reset to the zero vector after moving to a new tangent space.

# Constructor

    LowerTriangularAdaptivePoll(M, p=rand(M); kwargs...)

## Keyword arguments

* `basis=`[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`)
$(_kwargs([:retraction_method, :vector_transport_method, :X]))
"""
mutable struct LowerTriangularAdaptivePoll{
        P,
        T,
        F <: Real,
        V <: AbstractVector{F},
        M <: AbstractMatrix{F},
        I <: Int,
        B,
        VTM <: AbstractVectorTransportMethod,
        RM <: AbstractRetractionMethod,
    } <: AbstractMeshPollFunction
    base_point::P
    candidate::P
    poll_counter::I
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
        M::AbstractManifold,
        p = rand(M);
        basis::AbstractBasis = default_basis(M, typeof(p)),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(
            M
        ),
        X = zero_vector(M, p),
    )
    d = manifold_dimension(M)
    b_l = zeros(d)
    D_k = zeros(d, d + 1)
    return LowerTriangularAdaptivePoll(
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
"""
    is_successful(ltap::LowerTriangularAdaptivePoll)

Return whether the last [`LowerTriangularAdaptivePoll`](@ref) step was successful
"""
function is_successful(ltap::LowerTriangularAdaptivePoll)
    return ltap.last_poll_improved
end
"""
    get_descent_direction(ltap::LowerTriangularAdaptivePoll)

Return the direction of the last [`LowerTriangularAdaptivePoll`](@ref) that yields a descent of the cost.
If the poll was not successful, the zero vector is returned
"""
function get_descent_direction(ltap::LowerTriangularAdaptivePoll)
    return ltap.X
end
"""
    get_basepoint(ltap::LowerTriangularAdaptivePoll)

Return the base point of the tangent space, where the mash for the [`LowerTriangularAdaptivePoll`](@ref) is build in.
"""
function get_basepoint(ltap::LowerTriangularAdaptivePoll)
    return ltap.base_point
end
"""
    get_candidate(ltap::LowerTriangularAdaptivePoll)

Return the candidate of the last successful [`LowerTriangularAdaptivePoll`](@ref).
If the poll was unsuccessful, the base point is returned.
"""
function get_candidate(ltap::LowerTriangularAdaptivePoll)
    return ltap.candidate
end
"""
    update_basepoint!(M, ltap::LowerTriangularAdaptivePoll, p)

Update the base point of the [`LowerTriangularAdaptivePoll`](@ref).
This especially also updates the basis, that is used to build a (new) mesh.
"""
function update_basepoint!(M, ltap::LowerTriangularAdaptivePoll{P}, p::P) where {P}
    vector_transport_to!(
        M, ltap.X, ltap.base_point, ltap.X, p, ltap.vector_transport_method
    )
    copyto!(M, ltap.base_point, p)
    # reset candidate as well
    copyto!(M, ltap.candidate, ltap.base_point)
    return ltap
end
function show(io::IO, ltap::LowerTriangularAdaptivePoll)
    s = """LowerTriangularAdaptivePoll
    with
    * basis on the tangent space: $(ltap.basis)
    * retraction_method:          $(ltap.retraction_method)
    * vector_transport_method:    $(ltap.vector_transport_method)
    """
    return print(io, s)
end
function (ltap::LowerTriangularAdaptivePoll)(
        amp::AbstractManoptProblem,
        mesh_size::Real;
        scale_mesh::Real = 1.0,
        max_stepsize::Real = Inf,
    )
    M = get_manifold(amp)
    n = manifold_dimension(M)
    l = -log(4, mesh_size)
    S = (-2^l + 1):(2^l - 1)
    if ltap.poll_counter <= l # we did not yet generate a b_l on this scale
        ltap.poll_counter += 1
        # A random index ι
        ltap.random_index = rand(1:n)
        # generate a random b_l vector
        rand!(ltap.random_vector, S)
        ltap.random_vector[ltap.random_index] = rand((-2^l, 2^l))
    end #otherwise we already created ltap.random_vector for this mesh size
    # Generate L lower triangular, (n-1)x(n-1) in M
    for i in 1:(n - 1)
        for j in 1:(n - 1)
            if i < j
                ltap.mesh[i, j] = rand(S)
            elseif i == j
                ltap.mesh[i, j] = rand((-2^l, 2^l))
            else
                ltap.mesh[i, j] = 0
            end
        end
    end
    # Shift to construct n × n matrix B
    # (bottom left)
    if n > 1
        ltap.mesh[(ltap.random_index + 1):n, (1:n)] = ltap.mesh[
            (ltap.random_index):(n - 1), (1:n),
        ]
        # zero row above
        ltap.mesh[ltap.random_index, (1:n)] .= 0
    end
    # second to last column: random vector
    ltap.mesh[:, n] .= ltap.random_vector
    # set last column to minus the sum.
    ltap.mesh[:, n + 1] .= -1 .* sum(ltap.mesh[:, 1:n]; dims = 2)
    # Permute B (first n columns)
    ltap.mesh[:, 1:n] .= ltap.mesh[:, randperm(n)]
    # look for best
    ltap.last_poll_improved = false
    i = 0
    c = get_cost(amp, ltap.base_point)
    while (i < (n + 1)) && !(ltap.last_poll_improved)
        i = i + 1 # runs for the last time for i=n+1 and hence the sum.
        # get vector – scale mesh
        get_vector!(
            M,
            ltap.X,
            ltap.base_point,
            mesh_size * scale_mesh .* ltap.mesh[:, i],
            ltap.basis,
        )
        # shorten if necessary
        ltap_X_norm = norm(M, ltap.base_point, ltap.X)
        if ltap_X_norm > max_stepsize
            ltap.X .*= max_stepsize / ltap_X_norm
        end
        retract!(M, ltap.candidate, ltap.base_point, ltap.X, ltap.retraction_method)
        if get_cost(amp, ltap.candidate) < c
            # Keep old point and search direction, since the update will come later only
            # copyto!(M, ltap.base_point, ltap.candidate)
            ltap.last_poll_improved = true
            # this also breaks while
        end
    end
    # clear temp vector if we did not improve – set to zero vector and “clear” candidate.
    if !(ltap.last_poll_improved)
        zero_vector!(M, ltap.X, ltap.base_point)
        copyto!(M, ltap.candidate, ltap.base_point)
    end
    return ltap
end

#
# Specific Searches
#

"""
    DefaultMeshAdaptiveDirectSearch <: AbstractMeshSearchFunction

# Functor

    (s::DefaultMeshAdaptiveDirectSearch)(problem, mesh_size::Real, X; scale_mesh::Real=1.0, max_stepsize::Real=inf)

# Fields

* `q`: a temporary memory for a point on the manifold
* `X`: information to perform the search, e.g. the last direction found by poll.
* `last_search_improved::Bool` indicate whether the last search was successful, i.e. improved the cost.
$(_fields(:retraction_method))

# Constructor

    DefaultMeshAdaptiveDirectSearch(M::AbstractManifold, p=rand(M); kwargs...)

## Keyword arguments

$(_kwargs([:retraction_method, :X]))
"""
mutable struct DefaultMeshAdaptiveDirectSearch{P, T, RM <: AbstractRetractionMethod} <:
    AbstractMeshSearchFunction
    p::P
    q::P
    X::T
    last_search_improved::Bool
    retraction_method::RM
end
function DefaultMeshAdaptiveDirectSearch(
        M::AbstractManifold,
        p = rand(M);
        X = zero_vector(M, p),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M),
    )
    return DefaultMeshAdaptiveDirectSearch(p, copy(M, p), X, false, retraction_method)
end
"""
    is_successful(dmads::DefaultMeshAdaptiveDirectSearch)

Return whether the last [`DefaultMeshAdaptiveDirectSearch`](@ref) was successful.
"""
function is_successful(dmads::DefaultMeshAdaptiveDirectSearch)
    return dmads.last_search_improved
end
"""
    get_candidate(dmads::DefaultMeshAdaptiveDirectSearch)

Return the last candidate a [`DefaultMeshAdaptiveDirectSearch`](@ref) found
"""
function get_candidate(dmads::DefaultMeshAdaptiveDirectSearch)
    return dmads.p
end
function show(io::IO, dmads::DefaultMeshAdaptiveDirectSearch)
    s = """DefaultMeshAdaptiveDirectSearch
      with
      * retraction_method: $(dmads.retraction_method)
    """
    return print(io, s)
end
function (dmads::DefaultMeshAdaptiveDirectSearch)(
        amp::AbstractManoptProblem,
        mesh_size::Real,
        p,
        X;
        scale_mesh::Real = 1.0,
        max_stepsize::Real = Inf,
    )
    M = get_manifold(amp)
    dmads.X .= (4 * mesh_size * scale_mesh) .* X

    dmads_X_norm = norm(M, p, dmads.X)
    if dmads_X_norm > max_stepsize
        dmads.X .*= max_stepsize / dmads_X_norm
    end
    retract!(M, dmads.q, p, dmads.X, dmads.retraction_method)
    dmads.last_search_improved = get_cost(amp, dmads.q) < get_cost(amp, p)
    if dmads.last_search_improved
        copyto!(M, dmads.p, dmads.q)
    end
    return dmads
end

"""
    MeshAdaptiveDirectSearchState <: AbstractManoptSolverState

# Fields

$(_fields(:p; add_properties = [:as_Iterate]))
* `mesh_size`: the current (internal) mesh size
* `scale_mesh`: the current scaling of the internal mesh size, yields the actual mesh size used
* `max_stepsize`: an upper bound for the longest step taken in looking for a candidate in either poll or search
* `poll_size`
$(_fields(:stopping_criterion; name = "stop"))
* `poll::`[`AbstractMeshPollFunction`]: a poll step (functor) to perform
* `search::`[`AbstractMeshSearchFunction`}(@ref) a search step (functor) to perform

"""
mutable struct MeshAdaptiveDirectSearchState{
        P,
        F <: Real,
        PT <: AbstractMeshPollFunction,
        ST <: AbstractMeshSearchFunction,
        SC <: StoppingCriterion,
    } <: AbstractManoptSolverState
    p::P
    mesh_size::F
    scale_mesh::F
    max_stepsize::F
    poll_size::F
    stop::SC
    poll::PT
    search::ST
end
function MeshAdaptiveDirectSearchState(
        M::AbstractManifold,
        p::P = rand(M);
        mesh_basis::B = default_basis(M, typeof(p)),
        scale_mesh::F = injectivity_radius(M) / 2,
        max_stepsize::F = injectivity_radius(M),
        stopping_criterion::SC = StopAfterIteration(500) | StopWhenPollSizeLess(1.0e-7),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(
            M, typeof(p)
        ),
        poll::PT = LowerTriangularAdaptivePoll(
            M,
            copy(M, p);
            basis = mesh_basis,
            retraction_method = retraction_method,
            vector_transport_method = vector_transport_method,
        ),
        search::ST = DefaultMeshAdaptiveDirectSearch(
            M, copy(M, p); retraction_method = retraction_method
        ),
    ) where {
        P,
        F,
        PT <: AbstractMeshPollFunction,
        ST <: AbstractMeshSearchFunction,
        SC <: StoppingCriterion,
        B <: AbstractBasis,
    }
    poll_s = manifold_dimension(M) * 1.0
    return MeshAdaptiveDirectSearchState{P, F, PT, ST, SC}(
        p, 1.0, scale_mesh, max_stepsize, poll_s, stopping_criterion, poll, search
    )
end
get_iterate(mads::MeshAdaptiveDirectSearchState) = mads.p

function status_summary(mads::MeshAdaptiveDirectSearchState; inline = false)
    i = get_count(mads, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    inline && (return "$(repr(mads)) – $(Iter) $(has_converged(mads) ? "(converged)" : "")")
    s = """
    # Solver state for `Manopt.jl`s mesh adaptive direct search
    $Iter
    ## Parameters
    * mesh_size: $(mads.mesh_size)
    * scale_mesh: $(mads.scale_mesh)
    * max_stepsize: $(mads.max_stepsize)
    * poll_size: $(mads.poll_size)
    * poll:\n  $(replace(repr(mads.poll), "\n" => "\n  ")[1:(end - 3)])
    * search:\n  $(replace(repr(mads.search), "\n" => "\n  ")[1:(end - 3)])

    ## Stopping criterion
    $(status_summary(mads.stop; inline = false))
    """
    return s
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
    return (inline ? "Poll step size s < $(c.threshold):$(_MANOPT_INDENT)" : "Stop when the poll step size is less than the threshold $(c.threshold)\n$(_MANOPT_INDENT)") * s
end
show(io::IO, c::StopWhenPollSizeLess) = print(io, "StopWhenPollSizeLess($(c.threshold))")
