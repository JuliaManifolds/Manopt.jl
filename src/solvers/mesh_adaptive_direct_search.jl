_doc_mads = """

    mesh_adaptive_direct_search(M, f, p=rand(M); kwargs...)
    mesh_adaptive_direct_search(M, mco::AbstractManifoldCostObjective, p=rand(M); kwargs..)
    mesh_adaptive_direct_search!(M, f, p; kwargs...)
    mesh_adaptive_direct_search!(M, mco::AbstractManifoldCostObjective, p; kwargs..)

The Mesh Adaptive Direct Search (MADS) algorithm minimizes an objective function ``f: $(_math(:Manifold))nifold))) → ℝ`` on the manifold `M`.
The algorithm constructs an implicit mesh in the tangent space ``$(_math(:TangentSpace)))`` at the current candidate ``p``.
Each iteration consists of a search step and a poll step.

The search step selects points from the implicit mesh and attempts to find an improved candidate solution that reduces the value of ``f``.
If the search step fails to generate an improved candidate solution, the poll step is performed.
It consists of a local exploration on the current implicit mesh in the neighbourhood of the current iterate.

# Input

$(_args([:M, :f, :p]))

# Keyword arguments

* `mesh_basis=`[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`):
  a basis to generate the mesh in. The mesh is generated in coordinates of this basis in every tangent space
* `max_stepsize=`$(_link(:injectivity_radius))`(M)`: a maximum step size to take.
  any vector generated on the mesh is shortened to this length to avoid leaving the injectivity radius,
* `poll::`[`AbstractMeshPollFunction`](@ref)`=`[`LowerTriangularAdaptivePoll`](@ref)`(M, copy(M,p))`:
  the poll function to use. The `mesh_basis` (as `basis`), `retraction_method`, and `vector_transport_method` are passed to this default as well.
$(_kwargs(:retraction_method))
* `scale_mesh=`$(_link(:injectivity_radius))`(M) / 4`: initial scaling of the mesh
* `search::`[`AbstractMeshSearchFunction`](@ref)`=`[`DefaultMeshAdaptiveDirectSearch`](@ref)`(M, copy(M,p))`:
  the search function to use. The `retraction_method` is passed to this default as well.
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenPollSizeLess`](@ref)`(1e-10)"))
$(_kwargs([:vector_transport_method, :X]))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_mads)"
mesh_adaptive_direct_search(M::AbstractManifold, args...; kwargs...)

function mesh_adaptive_direct_search(M::AbstractManifold, f, p = rand(M); kwargs...)
    mco = ManifoldCostObjective(f)
    return mesh_adaptive_direct_search(M, mco, p; kwargs...)
end
function mesh_adaptive_direct_search(
        M::AbstractManifold, mco::AbstractManifoldCostObjective, p = rand(M); kwargs...
    )
    keywords_accepted(mesh_adaptive_direct_search; kwargs...)
    q = copy(M, p)
    return mesh_adaptive_direct_search!(M, mco, q; kwargs...)
end
calls_with_kwargs(::typeof(mesh_adaptive_direct_search)) = (mesh_adaptive_direct_search!,)

@doc "$(_doc_mads)"
mesh_adaptive_direct_search!(M::AbstractManifold, args...; kwargs...)

function mesh_adaptive_direct_search!(M::AbstractManifold, f, p; kwargs...)
    mco = ManifoldCostObjective(f)
    return mesh_adaptive_direct_search!(M, mco, p; kwargs...)
end
function mesh_adaptive_direct_search!(
        M::AbstractManifold,
        mco::AbstractManifoldCostObjective,
        p;
        mesh_basis::B = default_basis(M, typeof(p)),
        scale_mesh::Real = injectivity_radius(M) / 4,
        max_stepsize::Real = injectivity_radius(M),
        stopping_criterion::StoppingCriterion = StopAfterIteration(500) |
            StopWhenPollSizeLess(1.0e-10),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, eltype(p)),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(
            M, eltype(p)
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
        kwargs..., #collect rest
    ) where {B <: AbstractBasis, PT <: AbstractMeshPollFunction, ST <: AbstractMeshSearchFunction}
    keywords_accepted(mesh_adaptive_direct_search!; kwargs...)
    dmco = decorate_objective!(M, mco; kwargs...)
    dmp = DefaultManoptProblem(M, dmco)
    madss = MeshAdaptiveDirectSearchState(
        M,
        p;
        mesh_basis = mesh_basis,
        scale_mesh = scale_mesh,
        max_stepsize = oftype(scale_mesh, max_stepsize),
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        vector_transport_method = vector_transport_method,
        poll = poll,
        search = search,
    )
    dmadss = decorate_state!(madss; kwargs...)
    solve!(dmp, dmadss)
    return get_solver_return(get_objective(dmp), dmadss)
end
calls_with_kwargs(::typeof(mesh_adaptive_direct_search!)) = (decorate_objective!, decorate_state!)

# Init already do a poll, since the first search requires a poll
function initialize_solver!(
        amp::AbstractManoptProblem, madss::MeshAdaptiveDirectSearchState
    )
    M = get_manifold(amp)
    # do one poll step
    madss.poll(
        amp, madss.mesh_size; scale_mesh = madss.scale_mesh, max_stepsize = madss.max_stepsize
    )
    if is_successful(madss.poll)
        copyto!(M, madss.p, get_candidate(madss.poll))
    end
    return madss
end
function step_solver!(amp::AbstractManoptProblem, madss::MeshAdaptiveDirectSearchState, k)
    M = get_manifold(amp)
    n = manifold_dimension(M)
    # search if the last poll or last search was successful
    if is_successful(madss.search) || is_successful(madss.poll)
        madss.search(
            amp,
            madss.mesh_size,
            get_candidate(madss.poll),
            get_descent_direction(madss.poll);
            scale_mesh = madss.scale_mesh,
            max_stepsize = madss.max_stepsize,
        )
    end
    # For successful search, copy over iterate - skip poll, but update base
    if is_successful(madss.search)
        copyto!(M, madss.p, get_candidate(madss.search))
        update_basepoint!(M, madss.poll, madss.p)
    else #search was not successful: poll
        update_basepoint!(M, madss.poll, madss.p)
        madss.poll(
            amp,
            madss.mesh_size;
            scale_mesh = madss.scale_mesh,
            max_stepsize = madss.max_stepsize,
        )
        # For successful poll, copy over iterate
        if is_successful(madss.poll)
            copyto!(M, madss.p, get_candidate(madss.poll))
        end
    end
    # If neither found a better candidate -> reduce step size, we might be close already!
    if !(is_successful(madss.poll)) && !(is_successful(madss.search))
        madss.mesh_size /= 4
    elseif madss.mesh_size < 0.25 # else
        madss.mesh_size *= 4  # Coarsen the mesh but not beyond 1
    end
    # Update poll size parameter
    madss.poll_size = n * sqrt(madss.mesh_size)
    return madss
end
