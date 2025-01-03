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
* `already_updated::Int`: a poll counter ``l_c``, to check whether the random_vector ``b_l`` was already created
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
function get_poll_point(ltap::LowerTriangularAdaptivePoll)
    return ltap.p
end
function (ltap::LowerTriangularAdaptivePoll)(
    amp::AbstractManoptProblem, mesh_size; scale_mesh=1.0, max_stepsize=inf
)
    M = get_manifold(amp)
    n = manifold_dimension(M)
    l = -log(4, mesh_size)
    S = (-2^l + 1):(2^l - 1)
    if ltap.poll_counter <= l # we did not yet generate a b_l on this scale
        ltap.poll_counter += 1
        ltap.random_index = rand(1:n)
        ltap.random_vector
        for i in 1:n
            if i == r
                ltap.random_vector[i] = rand([-2^l, 2^l])
            else
                ltap.random_vector[i] = rand(S)
            end
        end
    end #otherwise we already created ltap.randomvector for this mesh size
    # Generate L lower triangular, (n-1)x(n-1) in M
    for i in 1:(n - 1)
        for j in n - 1
            poll.mesh[i, j] = (j > i) ? 0.0 : ((i == j) ? rand([-2^l, 2^l]) : rand(S))
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
        get_vector!(M, ltap.X, p, scale_mesh .* ltap.mesh[:, i], ltap.basis)
        # shorten if necessary
        if norm(M, ltap, p, ltap.X) > max_stepsize
            ltap.X = max_stepsize & norm(M, ltap, p, ltap.X) * ltap.X
        end
        retract!(M, ltap.q, ltap.p, ltap.X, ltap.retraction_method)
        if get_cost(amp, ltap.q) < c
            copyto!(M, ltap.p, ltap, q)
            ltap.last_poll_improved = true
            # this also breaks while
        end
    end
    # clear temp vector if we did not improve.
    !(ltap.last_poll_improved) && (zero_vector!(M, ltap.X, p))
    return ltap
end
"""
    DefaultMeshAdaptiveDirectSearch <: AbstractMeshSearchFunction

# Functor

    (s::DefaultMeshAdaptiveDirectSearch)(problem, mesh_size, X; scale_mesh=1.0, max_stepsize=inf)


# Fields

* `q`: a temporary memory for a point on the manifold
* `X`: the search direction
* `last_seach_improved::Bool` indicate whether the last search was succesfull, i.e. improved the cost.
$(_var(:Field, :retraction_method))
"""
mutable struct DefaultMeshAdaptiveDirectSearch{P,T} <: AbstractMeshSearchFunction
    p::P
    X::T
    last_search_improved::Bool
    retraction_method::RM
end
function DefaultMeshAdaptiveDirectSearch(
    M, p=rand(M); X=zero_vector(M, p), retraction_method=default_retaction_method(M)
)
    return DefaultMeshAdaptiveDirectSearch(p, X, false, retracttion_method)
end
function get_search_success(search!::DefaultMeshAdaptiveDirectSearch)
    return search!.last_search_improved
end
function get_search_point(search!::DefaultMeshAdaptiveDirectSearch)
    return search!.last_search_improved
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

# TODO: lower_triangular_mesh_adaptive_direct_search highlevel interface

# Init already do a poll, since the first search requires a poll
function initialize_solver!(
    amp::AbstractManoptProblem, madss::MeshAdaptiveDirectSearchState
)
    # do one poll step
    return madss.poll(
        amp, madss.mesh_size; scale_mesh=madss.scale_mesh, max_stepsize=madss.max_stepsize
    )
end
# TODO: step_solver to call search, poll and update both sizes.
function step_solver!(amp::AbstractManoptProblem, madss::MeshAdaptiveDirectSearchState, k)
    M = get_manifolds(amp)
    n = manifold_dimension(M)
    copyto!(M, madss.q, madss.p) # copy p to store previous q
    # search
    # TODO: update search if we moved -> PT X
    # update_search!(poll, madss.p)
    madss.search(
        amp,
        madss.mesh_size,
        madss.mesh_size,
        get_poll_direction(madss.poll);
        scale_mesh=madss.scale_mesh,
        max_stepsize=madss.max_stepsize,
    )
    # For sucesful search, copy over iterate - skip poll
    if get_search_success(madss.seachr)
        copyto!(M, madss.p, get_search_point(madss.search))
    else #search was not sucessful: poll
        #TODO: update poll basis -> vector transport from poll.p to madss.p
        # * at least poll.X
        # * better also matrix
        # * probably also basis if cached
        #
        # update_poll!(poll, madss.p)
        #
        madss.poll(
            amp,
            madss.mesh_size;
            scale_mesh=madss.scale_mesh,
            max_stepsize=madss.max_stepsize,
        )
        # For succesfull poll, copy over iterate
        if get_poll_success(madss.poll)
            copyto!(M, madss.p, get_poll_point(madss.search))
        end
    end
    # If neither found a better candidate -> reduce step size, we might be close already!
    if !(get_poll_success(madss.poll)) && !(get_search_success(madss.search))
        madss.mesh_size /= 4
    elseif madss.mesh_size < 0.25 # else
        madss.mesh_size *= 4  # Coarsen the mesh but not beyond 1
    end
    # Update poll size parameter
    return madss.poll_size = n * sqrt(madss.mesh_size)
end
