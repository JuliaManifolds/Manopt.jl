# Helpers for general manifold functions

function add_vector!(M::AbstractManifold, X, p, c, basis::AbstractBasis)
    Y = get_vector(M, p, c, basis)
    X .+= Y
    return X
end
function add_vector!(M::ProductManifold, X, p, c, basis::AbstractBasis)
    dims = map(manifold_dimension, M.manifolds)
    @assert length(c) == sum(dims)
    dim_ranges = ManifoldsBase._get_dim_ranges(dims)
    tc = map(dr -> (@inbounds view(c, dr)), dim_ranges)
    ts = ManifoldsBase.ziptuples(
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        tc,
    )
    map(ts) do t
        return add_vector!(t..., basis)
    end
    return X
end
function add_coordinates!(M::AbstractManifold, c, p, X, basis::AbstractBasis)
    cX = get_coordinates(M, p, X, basis)
    c .+= cX
    return c
end
