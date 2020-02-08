"""
`Manopt.jl` – Optimization on Manifolds in Julia.
"""
module Manopt
    using SimpleTraits
    using Markdown
    import Random: rand
    import ManifoldsBase:
        AbstractVectorTransportMethod,
        ParallelTransport,
        Manifold,
        distance,
        exp,
        exp!,
        log,
        injectivity_radius,
        inner,
        geodesic,
        norm,
        shortest_geodesic,
        vector_transport_to,
        vector_transport_to!,
        zero_tangent_vector,
        zero_tangent_vector!
    import Manifolds:
        _read,
        _write,
        Sphere,
        Euclidean,
        Circle,
        SymmetricPositiveDefinite,
        AbstractPowerManifold,
        PowerManifold,
        ProductManifold,
        ProductRepr,
        DiagonalizingOrthonormalBasis,
        ArrayPowerRepresentation,
        NestedPowerRepresentation
    import Manifolds:
        _read,
        _write,
        get_basis,
        get_coordinates,
        get_vector,
        get_vectors,
        get_iterator,
        mean,
        representation_size,
        sym_rem,
        ℝ,
        ℂ,
        ×,
        ^
    import Random:
        randperm

    using LinearAlgebra: Symmetric, eigen, svd
    """
        mid_point(M, p, q, x)

    Compute the mid point between p and q. If there is more than one mid point
    of (not neccessarily miniizing) geodesics (i.e. on the sphere), the one nearest
    to z is returned.
    """
    mid_point(M::MT, p, q, x) where {MT <: Manifold} = mid_point(M, p, q)

    mid_point!(M::MT, y, p, q, x) where {MT <: Manifold} = mid_point!(M, y, p, q)

    """
        mid_point(M, p, q)

    Compute the (geodesic) mid point of the two points `p` and `q` on the
    manfold `M`. If the geodesic is not unique, either a deterministic choice is taken or
    an error is raised depending on the manifold. For the deteministic choixe, see
    [`mid_point(M, p, q, x)`](@ref), the mid point closest to a third point
    `x`.
    """
    mid_point(M::MT, p, q) where {MT <: Manifold} = exp(M, p, log(M, p, q), 0.5)
    mid_point!(M::MT, y, p, q) where {MT <: Manifold} = exp!(M, y, p, log(M, p, q), 0.5)

    opposite(::Sphere,x) = -x
    opposite(::Circle{ℝ},x) = sym_rem(x+π)
    opposite(::Circle{ℂ},x) = -x

    rand(::Manifold) = error("Not yet implemented.")

    include("plans/plan.jl")
    # Functions
    include("functions/adjointDifferentials.jl")
    include("functions/costFunctions.jl")
    include("functions/differentials.jl")
    include("functions/gradients.jl")
    include("functions/jacobiFields.jl")
    include("functions/proximalMaps.jl")
    # ...solvers (1) general framework
    include("solvers/solver.jl")
    # ...solvers (2) specific solvers
    include("solvers/cyclicProximalPoint.jl")
    include("solvers/DouglasRachford.jl")
    include("solvers/NelderMead.jl")
    include("solvers/steepestDescent.jl")
    include("solvers/truncatedConjugateGradient.jl")
    include("solvers/trustRegions.jl")
    include("solvers/subGradientMethod.jl")
    include("solvers/debugSolver.jl")
    include("solvers/recordSolver.jl")
    include("plots/SpherePlots.jl")
    include("helpers/errorMeasures.jl")
    include("helpers/exports/Asymptote.jl")
    include("data/artificialDataFunctions.jl")

export
    ×, ^, ℝ, ℂ
export
    AdjDpGeo,
    AdjDqGeo,
    AdjDpExp,
    AdjDpExp,
    AdjDpLog,
    AdjDqLog,
    AdjDforwardLogs,
    distance,
    exp,
    geodesic,
    getGradient,
    getSubGradient,
    getCost,
    getProximalMap,
    log,
    mid_point,
    shortest_geodesic,
    sym_rem,
    vector_transport_to,
    vector_transport_to!,
    zero_tangent_vector,
    zero_tangent_vector!
export
    Problem,
    SubGradientProblem,
    GradientProblem,
    HessianProblem
export
    GradientDescentOptions,
    SubGradientMethodOptions
export
    Circle,
    Euclidean,
    Sphere,
    PowerManifold,
    PorductManifold
end
