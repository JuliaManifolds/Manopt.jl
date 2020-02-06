"""
`Manopt.jl` – Optimization on Manifolds in Julia.
"""
module Manopt
    using SimpleTraits
    using Markdown
    import Random: rand
    import ManifoldsBase:
        Manifold,
        distance,
        exp,
        log,
        inner,
        geodesic,
        norm
    import Manifolds:
        Sphere,
        Euclidean,
        Circle,
        PowerManifold,
        ProductManifold,
        ProductRepr,
        DiagonalizingOrthonormalBasis,
        mean,
        sym_rem,
        ℝ,
        ℂ
    import Random:
        randperm

    """
        midPoint(M, p, q, x)

    Compute the mid point between p and q. If there is more than one mid point
    of (not neccessarily miniizing) geodesics (i.e. on the sphere), the one nearest
    to z is returned.
    """
    midPoint(M::MT, p, q, x) where {MT <: Manifold} = midPoint(M, p, q)

    midPoint!(M::MT, y, p, q, x) where {MT <: Manifold} = midPoint!(M, y, p, q)

    """
        midPoint(M, p, q)

    Compute the (geodesic) mid point of the two points `p` and `q` on the
    manfold `M`. If the geodesic is not unique, either a deterministic choice is taken or
    an error is raised depending on the manifold. For the deteministic choixe, see
    [`midPoint(M, p, q, x)`](@ref), the mid point closest to a third point
    `x`.
    """
    midPoint(M::MT, p, q) where {MT <: Manifold} = exp(M, p, log(M, p, q), 0.5)
    midPoint!(M::MT, y, p, q) where {MT <: Manifold} = exp!(M, y, p, log(M, p, q), 0.5)

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
    # extended metasolvers
    include("solvers/debugSolver.jl")
    include("solvers/recordSolver.jl")
    # Plots
    include("plots/SpherePlots.jl")
    # helpers
    include("helpers/errorMeasures.jl")
    # Exports
    include("helpers/exports/Asymptote.jl")
    # data
    include("data/artificialDataFunctions.jl")

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
    getGradient,
    getCost,
    getProximalMap,
    log,
    Problem,
    HessianProblem
end
