
_doc_mads = """

    mesh_adaptive_direct_search(M, f, p=rand(M); kwargs...)
    mesh_adaptive_direct_search!(M, f, p; kwargs...)


# Keyword arguments

* `basis=`[`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`)
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :vector_transport_method))
$(_var(:Keyword, :X))
"""

@doc "$(_doc_mads)"
mesh_adaptive_direct_search(M::AbstractManifold, args...; kwargs...)

function mesh_adaptive_direct_search(M::AbstractManifold, f; kwargs...)
    return mesh_adaptive_direct_search(M, f, rand(M); kwargs...)
end
function mesh_adaptive_direct_search(M::AbstractManifold, f, p; kwargs...)
    mco = ManifoldCostObjective(f)
    return mesh_adaptive_direct_search(M, mco; kwargs...)
end
function mesh_adaptive_direct_search(
    M::AbstractManifold, mco::AbstractManifoldCostObjective, p; kwargs...
)
    q = copy(M, p)
    return mesh_adaptive_direct_search!(M, mco, q; kwargs...)
end

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
    mesh_basis::B=DefaultOrthonormalBasis(),
    scale_mesh=injectivity_radius(M) / 2,
    max_stepsize=injectivity_radius(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(500) |
                                          StopWhenPollSizeLess(1e-10),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, eltype(p)),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M, eltype(p)
    ),
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
    kwargs..., #collect rest
) where {B<:AbstractBasis,PT<:AbstractMeshPollFunction,ST<:AbstractMeshSearchFunction}
    dmco = decorate_objective!(M, mco; kwargs...)
    dmp = DefaultManoptProblem(M, dmco)
    madss = MeshAdaptiveDirectSearchState(
        M;
        mesh_basis=mesh_basis,
        scale_mesh=scale_mesh,
        max_stepsize=max_stepsize,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
        poll=poll,
        search=search,
    )
    dmadss = decorate_state!(madss; kwargs...)
    solve!(dmp, dmadss)
    return get_solver_return(get_objective(dmp), dmadss)
end

# Init already do a poll, since the first search requires a poll
function initialize_solver!(
    amp::AbstractManoptProblem, madss::MeshAdaptiveDirectSearchState
)
    M = get_manifold(amp)
    # do one poll step
    madss.poll(
        amp, madss.mesh_size; scale_mesh=madss.scale_mesh, max_stepsize=madss.max_stepsize
    )
    if get_poll_success(madss.poll)
        copyto!(M, madss.p, get_poll_best_candidate(madss.poll))
    end
    return madss
end
function step_solver!(amp::AbstractManoptProblem, madss::MeshAdaptiveDirectSearchState, k)
    M = get_manifold(amp)
    n = manifold_dimension(M)
    # search if the last poll or last search was sucessful
    if get_search_success(madss.search) || get_poll_success(madss.poll)
        madss.search(
            amp,
            madss.mesh_size,
            get_poll_best_candidate(madss.poll),
            get_poll_direction(madss.poll);
            scale_mesh=madss.scale_mesh,
            max_stepsize=madss.max_stepsize,
        )
    end
    # For succesful search, copy over iterate - skip poll, but update base
    if get_search_success(madss.search)
        copyto!(M, madss.p, get_search_point(madss.search))
        update_poll_basepoint!(M, madss.poll, madss.p)
    else #search was not sucessful: poll
        update_poll_basepoint!(M, madss.poll, madss.p)
        madss.poll(
            amp,
            madss.mesh_size;
            scale_mesh=madss.scale_mesh,
            max_stepsize=madss.max_stepsize,
        )
        # For succesfull poll, copy over iterate
        if get_poll_success(madss.poll)
            copyto!(M, madss.p, get_poll_best_candidate(madss.poll))
        end
    end
    # If neither found a better candidate -> reduce step size, we might be close already!
    if !(get_poll_success(madss.poll)) && !(get_search_success(madss.search))
        madss.mesh_size /= 4
    elseif madss.mesh_size < 0.25 # else
        madss.mesh_size *= 4  # Coarsen the mesh but not beyond 1
    end
    # Update poll size parameter
    madss.poll_size = n * sqrt(madss.mesh_size)
    return madss
end
