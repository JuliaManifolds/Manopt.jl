"""
    LowerTriangularAdaptivePoll <: AbstractMesh

Generate a mesh based on Section 6 and 7 of [Deisigmeyer:2007](@ref)

# Fields
* `p::P`: a point on the manifold, where the mesh is build in the tangent space
* `q::P`: a memory for a new point/candidate
* `already_updated::Int`: a poll counter ``l_c``, to check whether the random_vector ``b_l`` was already created
* `random_vector`: a ``d``-dimensional random vector ``b_l```
* `random_index`: a random index ``ι``
* `mesh`
* `basis`: a basis of the current tangent space with respect to which the mesh is stored
* `last_poll::T` the last successful poll direction stored as a tangent vector.
  initiliased to the zero vector and reset to the zero vector after moving to a new tangent space.
* `vector_transport_method`:
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
    already_updated::I
    random_vector::V
    random_index::I
    mesh::M
    basis::B
    last_poll::T
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
    D_k = zeros(d, d)
    last_poll = zero_vector(M, p)
    return LowerTriangularAdaptiveMesh(
        p,
        copy(M, p),
        0,
        b_l,
        0,
        D_k,
        basis,
        last_poll,
        false,
        retraction_method,
        vector_transport_method,
    )
end
function get_poll_success(poll!::LowerTriangularAdaptivePoll)
    return poll!.last_poll_improved
end
function (poll!::LowerTriangularAdaptivePoll)(amp::AbstractManoptProblem, stepsize)
    return M = get_manifold(amp)
    # Implement the code from Dreisigmeyer p. 16/17 about mesh generation
end
"""
    DefaultSearch <: AbstractMeshSearchFunction

# Fields

* `q`: a temporary memory for a point on the manifold
* `X`: the search direction
* `last_seach_improved::Bool` indicate whether the last search was succesfull, i.e. improved the cost.
* `retraction_method` – a method to perform the retractiom
"""
mutable struct DefaultMeshAdaptiveDirectSearch{P,T} <: AbstractMeshSearchFunction
    q::P
    X::T
    last_seach_improved::Bool
end
function get_search_success(search!::DefaultMeshAdaptiveDirectSearch)
    return search!.last_seach_improved
end
function (search!::DefaultMeshAdaptiveDirectSearch)(amp::AbstractManoptProblem, mesh_size)
    return M = get_manifold(amp)
    # Implement the code from Dreisigmeyer p. 17 about search generation
end

# TODO: lower_triangular_mesh_adaptive_direct_search highlevel interface

# TODO: Init solver: Do a first poll already? Probably good idea.
# TODO: step_solver to call search, poll and update both sizes.