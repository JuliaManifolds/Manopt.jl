function Manopt.ChambollePockState(M::AbstractManifold; kwargs...)
    return ChambollePockState(M, TangentBundle(M); kwargs...)
end
