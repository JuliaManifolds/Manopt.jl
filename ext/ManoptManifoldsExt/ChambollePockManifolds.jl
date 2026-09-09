function Manopt.ChambollePockState(
        M::AbstractManifold; N::AbstractManifold = TangentBundle(M), kwargs...,
    )
    return ChambollePockState(M, N; kwargs...)
end
