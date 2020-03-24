"""
`Manopt.jl` – Optimization on Manifolds in Julia.
"""
module Manopt
    using Colors
    using ColorSchemes
    using ColorTypes
    using SimpleTraits
    using Markdown
    using LinearAlgebra
    import Random:
        rand,
        randperm
    import Base:
        copy
    import ManifoldsBase:
        ℝ,
        ℂ,
        ×,
        ^,
        AbstractVectorTransportMethod,
        ParallelTransport,
        Manifold,
        distance,
        exp,
        exp!,
        log,
        log!,
        injectivity_radius,
        inner,
        geodesic,
        manifold_dimension,
        norm,
        project,
        project!,
        retract,
        retract!,
        shortest_geodesic,
        vector_transport_to,
        vector_transport_to!,
        zero_tangent_vector,
        zero_tangent_vector!,
        DiagonalizingOrthonormalBasis,
        get_basis,
        get_coordinates,
        get_vector,
        get_vectors,
        representation_size
    using Manifolds:
        AbstractPowerManifold,
        PowerManifold
    using Manifolds: #temporary for random
        Circle,
        Euclidean,
        Grassmann,
        Hyperbolic,
        ProductManifold,
        Rotations,
        SymmetricPositiveDefinite,
        Stiefel,
        Sphere
    using Manifolds: # Wishlist for Base
        NestedPowerRepresentation,
        mean,
        median,
        get_iterator,
        _read,
        _write

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

    reflect(M::Manifold, pr::Function, x) = reflect(M::Manifold, pr(x), x)
    reflect(M::Manifold, p, x) = exp(M, p, -log(M, p, x))

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

    include("random.jl")

export ×, ^, ℝ, ℂ

export AbstractOptionsAction, StoreOptionsAction
export hasStorage, getStorage, updateStorage!

export βDxg, βDpExp, βDXExp, βDpLog, βDqLog
export adjointJacobiField
export AdjDpGeo, AdjDqGeo, AdjDpExp, AdjDpExp, AdjDpLog, AdjDqLog, AdjDforwardLogs
export asyExportS2Signals, asyExportS2Data, asyExportSPDData
export costL2TV, costL2TVTV2, costL2TV2, costTV, costTV2, costIntrICTV12
export DpGeo, DqGeo, DpExp,DξExp, DqLog, DyLog, DforwardLogs
export jacobiField
export gradTV, gradTV2, gradIntrICTV12, forwardLogs, gradDistance
export getCost, getGradient, getSubGradient, getProximalMap, getOptions, getInitialStepsize
export getHessian, approxHessianFD
export meanSquaredError, meanAverageError
export proxDistance, proxTV, proxParallelTV, proxTV2, proxCollaborativeTV
export random_point, random_tangent
export stopIfResidualIsReducedByFactor, stopIfResidualIsReducedByPower, stopWhenCurvatureIsNegative, stopWhenTrustRegionIsExceeded

export DebugOptions, DebugAction, DebugGroup, DebugEntry, DebugEntryChange, DebugEvery
export DebugChange, DebugIterate, DebugIteration, DebugDivider
export DebugCost, DebugStoppingCriterion, DebugFactory, DebugActionFactory
export DebugGradient, DebugGradientNorm, DebugStepsize

export RecordGradient, RecordGradientNorm, RecordStepsize

export CostProblem, Problem, SubGradientProblem, GradientProblem, HessianProblem

export GradientDescentOptions, HessianOptions, SubGradientMethodOptions, NelderMeadOptions
export TruncatedConjugateGradientOptions, TrustRegionsOptions

export StoppingCriterion, StoppingCriterionSet, Stepsize
export EvalOrder, LinearEvalOrder, RandomEvalOrder, FixedRandomEvalOrder
export Options, getOptions, getReason
export IsOptionsDecorator

end
