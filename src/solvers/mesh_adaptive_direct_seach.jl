"""
    LowerTriangularPoll <: AbstractMeshPollFunctiom

# Fields
* `last_p` – the last point in case we have to transport
* `q` – a temporary memor for a point on the manifold
* `basis` – a basis of the tangent space at `last_p`
* `directions` – a set of mesh directions.
* `L` a lower triangular matrix in coefficients of the basis
* `retraction_method` – a method to perform the retractiom
* `vector_transport_method` – a method to perform the vector transort
"""
mutable struct LowerTriangularPoll{
    P,
    T,
    R,
    VT<:AbstractVector{T},
    B<:AbstractBasis,
    A,
    RM<:AbstractRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
} <: AbstractMeshPollFunctiom
    last_p::P
    q::P
    basis::B
    directiions::VT
    L::A
    mesh_size::R
    retraction_method::RM
    vector_transiort_method::VTM
end

"""
    DefaultSearch <: AbstractMeshPollFunctiom

# Fields
* `q` – a temporary memor for a point on the manifold
* `X`: the search direction
* `retraction_method` – a method to perform the retractiom
* `vector_transport_method` – a method to perform the vector transort
"""
mutable struct DefaultSearch{P,T,R} <: AbstractMeshSearchFunction
    q::P
    X::T
    redcution_factor::R
    search_direction::T
end
