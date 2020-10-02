"""
`Manopt.jl` – Optimization on Manifolds in Julia.
"""
module Manopt
    using Colors
    using ColorSchemes
    using ColorTypes
    using Markdown
    using LinearAlgebra
    using Dates: Period, Nanosecond, value
    import Random:
        rand,
        randperm
    import Base:
        copy,
        identity
    import ManifoldsBase:
        ℝ,
        ℂ,
        ×,
        ^,
        AbstractVectorTransportMethod,
        AbstractRetractionMethod,
        ExponentialRetraction,
        ParallelTransport,
        Manifold,
        DefaultOrthonormalBasis,
        allocate_result,
        allocate_result_type,
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
        representation_size,
        is_manifold_point,
        is_tangent_vector
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
        ArrayPowerRepresentation,
        mean,
        median,
        get_iterator,
        _read,
        _write,
        power_dimensions,
        ArrayReshaper,
        prod_point,
        ShapeSpecification,
        ProductRepr,
        submanifold_components,
        submanifold_component,
        get_component,
        set_component!

    include("plans/plan.jl")
    # Functions
    include("functions/manifold.jl")
    include("functions/adjointDifferentials.jl")
    include("functions/costs.jl")
    include("functions/differentials.jl")
    include("functions/gradients.jl")
    include("functions/jacobiFields.jl")
    include("functions/proximalMaps.jl")
    # solvers general framework
    include("solvers/solver.jl")
    # specific solvers
    include("solvers/conjugate_gradient_descent.jl")
    include("solvers/cyclic_proximal_point.jl")
    include("solvers/DouglasRachford.jl")
    include("solvers/NelderMead.jl")
    include("solvers/steepest_descent.jl")
    include("solvers/truncated_conjugate_gradient_descent.jl")
    include("solvers/trust_regions.jl")
    include("solvers/subgradient.jl")
    include("solvers/debug_solver.jl")
    include("solvers/record_solver.jl")
    include("solvers/quasi_Newton.jl")
    include("plots/SpherePlots.jl")
    include("helpers/errorMeasures.jl")
    include("helpers/exports/Asymptote.jl")
    include("data/artificialDataFunctions.jl")

    include("random.jl")

export ×, ^, ℝ, ℂ
export AbstractOptionsAction, StoreOptionsAction
export has_storage, get_storage, update_storage!

export random_point, random_tangent, mid_point, mid_point!, reflect, sym_rem

export βdifferential_geodesic_startpoint, βdifferential_exp_basepoint
export βdifferential_exp_argument, βdifferential_log_basepoint, βdifferential_log_argument
export adjoint_differential_geodesic_startpoint, adjoint_differential_geodesic_endpoint
export adjoint_differential_exp_basepoint, adjoint_differential_exp_argument
export adjoint_differential_log_basepoint, adjoint_differential_log_argument
export adjoint_differential_forward_logs
export asymptote_export_S2_signals, asymptote_export_S2_data, asymptote_export_SPD
export render_asymptote
export costL2TV, costL2TVTV2, costL2TV2, costTV, costTV2, costIntrICTV12
export differential_geodesic_startpoint, differential_geodesic_endpoint
export differential_exp_basepoint, differential_exp_argument
export differential_log_basepoint, differential_log_argument, differential_forward_logs
export jacobi_field, adjoint_Jacobi_field
export ∇TV, ∇TV2, ∇intrinsic_infimal_convolution_TV12, forward_logs, ∇distance
export get_cost, get_gradient, get_subgradient, getProximalMap, get_options, get_initial_stepsize
export getHessian, approxHessianFD
export meanSquaredError, meanAverageError
export prox_distance, prox_TV, prox_parallel_TV, prox_TV2, prox_collaborative_TV
export random_point, random_tangent
export stopIfResidualIsReducedByFactor, stopIfResidualIsReducedByPower, StopWhenCurvatureIsNegative, StopWhenTrustRegionIsExceeded
export StopAfterIteration, StopWhenChangeLess, StopWhenGradientNormLess, StopWhenCostLess
export StopAfter, StopWhenAll, StopWhenAny
export get_active_stopping_criteria, get_stopping_criteria, get_reason

export DebugOptions, DebugAction, DebugGroup, DebugEntry, DebugEntryChange, DebugEvery
export DebugChange, DebugIterate, DebugIteration, DebugDivider
export DebugCost, DebugStoppingCriterion, DebugFactory, DebugActionFactory
export DebugGradient, DebugGradientNorm, DebugStepsize

export RecordGradient, RecordGradientNorm, RecordStepsize

export CostProblem, Problem, SubGradientProblem, GradientProblem, HessianProblem

export NelderMead, steepest_descent, subgradient_method, truncated_conjugate_gradient_descent, trust_regions
export cyclic_proximal_point, conjugate_gradient_descent

export DebugGradient, DebugGradientNorm, DebugStepsize

export Options, get_options
export is_options_decorator, dispatch_options_decorator

export RLBFGSOptions, QuasiNewtonOptions
export quasi_Newton, WolfePowellLineseach, StrongWolfePowellLineseach, WolfePowellLineseachHuang

export
    ConjugateGradientDescentOptions,
    GradientDescentOptions,
    HessianOptions,
    SubGradientMethodOptions,
    NelderMeadOptions,
    TruncatedConjugateGradientOptions,
    TrustRegionsOptions

export DirectionUpdateRule,
    SteepestDirectionUpdateRule,
    HeestenesStiefelCoefficient,
    FletcherReevesCoefficient,
    PolakRibiereCoefficient,
    ConjugateDescentCoefficient,
    LiuStoreyCoefficient,
    DaiYuanCoefficient,
    HagerZhangCoefficient

export StoppingCriterion, StoppingCriterionSet, Stepsize
export EvalOrder, LinearEvalOrder, RandomEvalOrder, FixedRandomEvalOrder

export decorate_options
export initialize_solver!, step_solver!, get_solver_result, stop_solver!
export solve

export ConstantStepsize, DecreasingStepsize
export Linesearch, ArmijoLinesearch
export get_stepsize, get_initial_stepsize, get_last_stepsize

end
