import Base: deleteat!, push!

mutable struct BundleStruct{
    I<:Integer,Is<:Vector{<:I},P,Ps<:Vector{<:P},T,Ts<:Vector{<:T}
}
    points::Ps
    vectors::Ts
    indices::Is
    size::I
    function BundleStruct(
        M::AM, p::P, X::T, size::I
    ) where {AM<:AbstractManifold,I<:Integer,P,T}
        points = [copy(M, p) for _ in 1:size]
        vectors = [copy(M, p, X) for _ in 1:size]
        indices = [1]
        size = size
        return new{I,P,T,typeof(points),typeof(vectors),typeof(indices)}(
            points, vectors, indices, size
        )
    end
end
function push!(bs::BundleStruct, pX::Tuple, i::Int)
    copyto!(bs.points[bs.indices[end]], pX[1])
    copyto!(bs.vectors[bs.indices[end]], pX[2])
    push!(bs.indices, i)
    return bs
end
function deleteat!(bs::BundleStruct, i::Int)
    bs.indices[i] = 0
    return bs
end

@doc raw"""
    BundleMethodState <: AbstractManoptSolverState
stores option values for a [`bundle_method`](@ref) solver

# Fields

* `bundle` - bundle that collects each iterate with the computed subgradient at the iterate
* `index_set` - the index set that keeps track of the strictly positive convex coefficients of the subproblem
* `inverse_retraction_method` - the inverse retraction to use within
* `lin_errors` - linearization errors at the last serious step
* `m` - the parameter to test the decrease of the cost
* `p` - current iterate
* `p_last_serious` - last serious iterate
* `retraction_method` – the retraction to use within
* `stop` – a [`StoppingCriterion`](@ref)
* `vector_transport_method` - the vector transport method to use within
* `X` - (`zero_vector(M, p)`) the current element from the possible subgradients at
    `p` that was last evaluated.
* `ξ` - the stopping parameter given by ξ = -\norm{g}^2 - ε

# Constructor

BundleMethodState(M::AbstractManifold, p; kwargs...)

with keywords for all fields above besides `p_last_serious` which obtains the same type as `p`.
You can use e.g. `X=` to specify the type of tangent vector to use

"""
mutable struct BundleMethodState{
    R,
    P,
    T,
    A<:Vector{<:R},
    B<:Vector{Tuple{<:P,<:T}},
    C<:Vector{T},
    I,
    D<:Vector{<:I},
    IR<:AbstractInverseRetractionMethod,
    TR<:AbstractRetractionMethod,
    TSC<:StoppingCriterion,
    VT<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState where {R<:Float64, P,T, I<:Int64}
    bundle::B
    bundle_size::I
    indices::D
    indices_ref::D
    inverse_retraction_method::IR
    j::I
    lin_errors::A
    p::P
    p_last_serious::P
    p0::P
    q0::P
    X::T
    retraction_method::TR
    stop::TSC
    vector_transport_method::VT
    m::R
    ξ::R
    diam::R
    λ::A
    g::T
    ε::R
    transported_subgradients::C
    filter1::R
    filter2::R
    δ::R
    function BundleMethodState(
        M::TM,
        p::P;
        bundle_size::Integer=50,
        m::R=1e-2,
        diam::R=1.0,
        inverse_retraction_method::IR=default_inverse_retraction_method(M, typeof(p)),
        retraction_method::TR=default_retraction_method(M, typeof(p)),
        stopping_criterion::SC=StopWhenBundleLess(1e-8),
        X::T=zero_vector(M, p),
        vector_transport_method::VT=default_vector_transport_method(M, typeof(p)),
        filter1::R=eps(Float64),
        filter2::R=eps(Float64),
        δ::R=√2,
    ) where {
        IR<:AbstractInverseRetractionMethod,
        P,
        T,
        TM<:AbstractManifold,
        TR<:AbstractRetractionMethod,
        SC<:StoppingCriterion,
        VT<:AbstractVectorTransportMethod,
        R<:Real,
    }
        # Initialize indes set, bundle points, linearization errors, and stopping parameter
        bundle = [(copy(M, p), copy(M, p, X)) for _ in 1:bundle_size]
        indices = [0 for _ in 1:bundle_size]
        indices[1] = 1
        indices_ref = copy(indices)
        j = 1
        lin_errors = zeros(bundle_size)
        ξ = 0.0
        λ = [1.0]
        g = copy(M, p, X)
        ε = 0.0
        transported_subgradients = [copy(M, p, X)]
        return new{
            typeof(m),
            P,
            T,
            typeof(lin_errors),
            typeof(bundle),
            typeof(transported_subgradients),
            typeof(bundle_size),
            typeof(indices),
            IR,
            TR,
            SC,
            VT,
        }(
            bundle,
            bundle_size,
            indices,
            indices_ref,
            inverse_retraction_method,
            j,
            lin_errors,
            p,
            copy(M, p),
            copy(M, p),
            copy(M, p),
            X,
            retraction_method,
            stopping_criterion,
            vector_transport_method,
            m,
            ξ,
            diam,
            λ,
            g,
            ε,
            transported_subgradients,
            filter1,
            filter2,
            δ,
        )
    end
end
get_iterate(bms::BundleMethodState) = bms.p_last_serious
get_subgradient(bms::BundleMethodState) = bms.g
