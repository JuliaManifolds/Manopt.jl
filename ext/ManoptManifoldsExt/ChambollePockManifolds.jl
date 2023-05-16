function Manopt.ChambollePockState(
    M::AbstractManifold,
    m::P,
    n::Q,
    p::P,
    X::T;
    N::AbstractManifold=TangentBundle(M),
    kwargs...
) where {
    P,
    Q,
    T}
    return ChambollePockState(M, N, m, n, p, X; kwargs...)
end
