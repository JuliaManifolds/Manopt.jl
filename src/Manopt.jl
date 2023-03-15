"""
🏔️ Manopt.jl – Optimization on Manifolds in Julia.

- 📚 Documentation: https://manoptjl.org
- 📦 Repository: https://github.com/JuliaManifolds/Manopt.jl
- 💬 Discussions: https://github.com/JuliaManifolds/Manopt.jl/discussions
- 🎯 Issues: https://github.com/JuliaManifolds/Manopt.jl/issues
"""
module Manopt
import Base: &, copy, getindex, identity, setindex!, show, |
import LinearAlgebra: reflect!
import ManifoldsBase: mid_point, mid_point!

using ColorSchemes
using ColorTypes
using Colors
using DataStructures: CircularBuffer, capacity, length, push!, size
using Dates: Millisecond, Nanosecond, Period, canonicalize, value
using LinearAlgebra: Diagonal, I, eigen, eigvals, tril
using ManifoldDiff:
    adjoint_Jacobi_field,
    adjoint_Jacobi_field!,
    adjoint_differential_exp_argument,
    adjoint_differential_exp_argument!,
    adjoint_differential_exp_basepoint,
    adjoint_differential_exp_basepoint!,
    adjoint_differential_log_argument,
    adjoint_differential_log_argument!,
    adjoint_differential_log_basepoint,
    adjoint_differential_log_basepoint!,
    adjoint_differential_shortest_geodesic_endpoint,
    adjoint_differential_shortest_geodesic_endpoint!,
    adjoint_differential_shortest_geodesic_startpoint,
    adjoint_differential_shortest_geodesic_startpoint!,
    differential_exp_argument,
    differential_exp_argument!,
    differential_exp_basepoint,
    differential_exp_basepoint!,
    differential_log_argument,
    differential_log_argument!,
    differential_log_basepoint,
    differential_log_basepoint!,
    differential_shortest_geodesic_endpoint,
    differential_shortest_geodesic_endpoint!,
    differential_shortest_geodesic_startpoint,
    differential_shortest_geodesic_startpoint!,
    jacobi_field,
    jacobi_field!
using ManifoldsBase:
    AbstractBasis,
    AbstractDecoratorManifold,
    AbstractInverseRetractionMethod,
    AbstractManifold,
    AbstractPowerManifold,
    AbstractRetractionMethod,
    AbstractVectorTransportMethod,
    CachedBasis,
    DefaultManifold,
    DefaultOrthonormalBasis,
    DiagonalizingOrthonormalBasis,
    ExponentialRetraction,
    LogarithmicInverseRetraction,
    NestedPowerRepresentation,
    ParallelTransport,
    PowerManifold,
    ProjectionTransport,
    QRRetraction,
    ^,
    _read,
    _write,
    allocate,
    allocate_result,
    allocate_result_type,
    copy,
    copyto!,
    default_inverse_retraction_method,
    default_retraction_method,
    default_vector_transport_method,
    distance,
    embed_project,
    embed_project!,
    exp,
    exp!,
    geodesic,
    get_basis,
    get_component,
    get_coordinates,
    get_coordinates!,
    get_iterator,
    get_vector,
    get_vector!,
    get_vectors,
    injectivity_radius,
    inner,
    inverse_retract,
    inverse_retract!,
    is_point,
    is_vector,
    log,
    log!,
    manifold_dimension,
    norm,
    number_eltype,
    power_dimensions,
    project,
    project!,
    representation_size,
    requires_caching,
    retract,
    retract!,
    set_component!,
    shortest_geodesic,
    vector_transport_to,
    vector_transport_to!,
    zero_vector,
    zero_vector!,
    ×,
    ℂ,
    ℝ
using Markdown
using Printf
using Random: shuffle!
using Requires
using SparseArrays
using StaticArrays
using Statistics: cor, cov, mean, std

include("plans/plan.jl")
# Functions
include("functions/bezier_curves.jl")
include("functions/adjoint_differentials.jl")
include("functions/costs.jl")
include("functions/differentials.jl")
include("functions/gradients.jl")
include("functions/proximal_maps.jl")
# solvers general framework
include("solvers/solver.jl")
# specific solvers
include("solvers/augmented_Lagrangian_method.jl")
include("solvers/bundle_method.jl")
include("solvers/ChambollePock.jl")
include("solvers/conjugate_gradient_descent.jl")
include("solvers/cyclic_proximal_point.jl")
include("solvers/DouglasRachford.jl")
include("solvers/exact_penalty_method.jl")
include("solvers/NelderMead.jl")
include("solvers/FrankWolfe.jl")
include("solvers/gradient_descent.jl")
include("solvers/LevenbergMarquardt.jl")
include("solvers/particle_swarm.jl")
include("solvers/primal_dual_semismooth_Newton.jl")
include("solvers/quasi_Newton.jl")
include("solvers/truncated_conjugate_gradient_descent.jl")
include("solvers/trust_regions.jl")
include("solvers/stochastic_gradient_descent.jl")
include("solvers/subgradient.jl")
include("solvers/debug_solver.jl")
include("solvers/record_solver.jl")
include("helpers/checks.jl")
include("helpers/errorMeasures.jl")
include("helpers/exports/Asymptote.jl")
include("data/artificialDataFunctions.jl")
include("deprecated.jl")

function __init__()
    @require Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e" begin
        using .Manifolds:
            Circle,
            Euclidean,
            Grassmann,
            GroupManifold,
            Hyperbolic,
            PositiveNumbers,
            ProductManifold,
            Rotations,
            SymmetricPositiveDefinite,
            Stiefel,
            Sphere,
            TangentBundle,
            TangentSpaceAtPoint,
            FixedRankMatrices,
            SVDMPoint,
            UMVTVector,
            ArrayPowerRepresentation,
            ProductRepr,
            submanifold_components,
            sym_rem,
            mean
        import Random: rand, randperm
        using LinearAlgebra: cholesky, det, diag, dot, Hermitian, qr, Symmetric, triu
        # adaptions for Nonmutating manifolds
        const NONMUTATINGMANIFOLDS = Union{Circle,PositiveNumbers,Euclidean{Tuple{}}}
        include("functions/manifold_functions.jl")
        include("functions/nonmutating_manifolds_functions.jl")
        include("plans/nonmutating_manifolds_plans.jl")
        include("plans/alternating_gradient_plan.jl")
        include("solvers/alternating_gradient_descent.jl")
        export mid_point, mid_point!, reflect, reflect!
        export AlternatingGradientDescentState
        export AlternatingGradient
        export alternating_gradient_descent, alternating_gradient_descent!
    end
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
        using .Plots
        include("helpers/check_plots.jl")
    end
    @require LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255" begin
        using .LineSearches
        include("ext/LineSearchesExt.jl")
    end
    @require QuadraticModels = "f468eda6-eac5-11e8-05a5-ff9e497bcd19" begin
        using .QuadraticModels: QuadraticModel
        @require RipQP = "1e40b3f8-35eb-4cd8-8edd-3e515bb9de08" begin
            using .RipQP: ripqp
            include("solvers/bundle_method_sub_solver.jl")
        end
    end

    return nothing
end
#
# General
export ℝ, ℂ, &, |
#
# Problems
export AbstractManoptProblem, DefaultManoptProblem, TwoManifoldProblem
#
# Objectives
export AbstractManifoldGradientObjective,
    AbstractManifoldCostObjective,
    AbstractManifoldObjective,
    AbstractPrimalDualManifoldObjective,
    ConstrainedManifoldObjective,
    NonlinearLeastSquaresObjective,
    ManifoldAlternatingGradientObjective,
    ManifoldCostGradientObjective,
    ManifoldCostObjective,
    ManifoldGradientObjective,
    ManifoldHessianObjective,
    ManifoldProximalMapObjective,
    ManifoldStochasticGradientObjective,
    ManifoldSubgradientObjective,
    PrimalDualManifoldObjective,
    PrimalDualManifoldSemismoothNewtonObjective,
    SimpleCacheObjective
#
# Evaluation & Problems - old
export AbstractEvaluationType, AllocatingEvaluation, InplaceEvaluation, evaluation_type
#
# AbstractManoptSolverState
export AbstractGradientSolverState,
    AbstractHessianSolverState,
    AbstractManoptSolverState,
    AbstractPrimalDualSolverState,
    AugmentedLagrangianMethodState,
    BundleMethodState,
    ChambollePockState,
    ConjugateGradientDescentState,
    CyclicProximalPointState,
    DouglasRachfordState,
    ExactPenaltyMethodState,
    FrankWolfeState,
    GradientDescentState,
    LevenbergMarquardtState,
    NelderMeadState,
    ParticleSwarmState,
    PrimalDualSemismoothNewtonState,
    RecordSolverState,
    StochasticGradientDescentState,
    SubGradientMethodState,
    TruncatedConjugateGradientState,
    TrustRegionsState

export FrankWolfeCost, FrankWolfeGradient
export NelderMeadSimplex
#
# Accessors and helpers for AbstractManoptSolverState
export linesearch_backtrack, default_stepsize
export get_cost, get_cost_function
export get_gradient, get_gradient_function, get_gradient!
export get_subgradient, get_subgradient!
export get_proximal_map,
    get_proximal_map!,
    get_state,
    get_initial_stepsize,
    get_iterate,
    get_gradients,
    get_gradients!,
    get_manifold,
    get_preconditioner,
    get_preconditioner!,
    get_primal_prox,
    get_primal_prox!,
    get_differential_primal_prox,
    get_differential_primal_prox!,
    get_dual_prox,
    get_dual_prox!,
    get_differential_dual_prox,
    get_differential_dual_prox!,
    set_gradient!,
    set_iterate!,
    set_manopt_parameter!,
    set_manopt_parameter!,
    set_manopt_parameter!,
    linearized_forward_operator,
    linearized_forward_operator!,
    adjoint_linearized_operator,
    adjoint_linearized_operator!,
    forward_operator,
    forward_operator!,
    get_objective
export set_manopt_parameter!
export get_hessian, get_hessian!, ApproxHessianFiniteDifference
export is_state_decorator, dispatch_state_decorator
export primal_residual, dual_residual
export get_constraints,
    get_inequality_constraint,
    get_inequality_constraints,
    get_equality_constraint,
    get_equality_constraints,
    get_grad_inequality_constraint,
    get_grad_inequality_constraint!,
    get_grad_inequality_constraints,
    get_grad_inequality_constraints!,
    get_grad_equality_constraint,
    get_grad_equality_constraint!,
    get_grad_equality_constraints,
    get_grad_equality_constraints!
export ConstraintType, FunctionConstraint, VectorConstraint
export AugmentedLagrangianCost, AugmentedLagrangianGrad, ExactPenaltyCost, ExactPenaltyGrad

export QuasiNewtonState, QuasiNewtonLimitedMemoryDirectionUpdate
export QuasiNewtonMatrixDirectionUpdate
export QuasiNewtonCautiousDirectionUpdate,
    BFGS, InverseBFGS, DFP, InverseDFP, SR1, InverseSR1
export InverseBroyden, Broyden
export AbstractQuasiNewtonDirectionUpdate, AbstractQuasiNewtonUpdateRule
export WolfePowellLinesearch,
    StrongWolfePowellLinesearch,
    operator_to_matrix,
    square_matrix_vector_product,
    WolfePowellBinaryLinesearch
export AbstractStateAction, StoreStateAction
export has_storage, get_storage, update_storage!
export objective_cache_factory
#
# Direction Update Rules
export DirectionUpdateRule,
    IdentityUpdateRule, StochasticGradient, AverageGradient, MomentumGradient, Nesterov
export DirectionUpdateRule,
    SteepestDirectionUpdateRule,
    HestenesStiefelCoefficient,
    FletcherReevesCoefficient,
    PolakRibiereCoefficient,
    ConjugateDescentCoefficient,
    LiuStoreyCoefficient,
    DaiYuanCoefficient,
    HagerZhangCoefficient,
    ConjugateGradientBealeRestart
#
# Solvers
export augmented_Lagrangian_method,
    augmented_Lagrangian_method!,
    bundle_method,
    bundle_method!,
    ChambollePock,
    ChambollePock!,
    conjugate_gradient_descent,
    conjugate_gradient_descent!,
    cyclic_proximal_point,
    cyclic_proximal_point!,
    DouglasRachford,
    DouglasRachford!,
    exact_penalty_method,
    exact_penalty_method!,
    Frank_Wolfe_method,
    Frank_Wolfe_method!,
    gradient_descent,
    gradient_descent!,
    LevenbergMarquardt,
    LevenbergMarquardt!,
    NelderMead,
    NelderMead!,
    particle_swarm,
    particle_swarm!,
    primal_dual_semismooth_Newton,
    quasi_Newton,
    quasi_Newton!,
    stochastic_gradient_descent,
    stochastic_gradient_descent!,
    subgradient_method,
    subgradient_method!,
    truncated_conjugate_gradient_descent,
    truncated_conjugate_gradient_descent!,
    trust_regions,
    trust_regions!
# Solver helpers
export decorate_state!, decorate_objective!
export initialize_solver!, step_solver!, get_solver_result, get_solver_return, stop_solver!
export solve!
export ApproxHessianFiniteDifference, ApproxHessianSymmetricRankOne, ApproxHessianBFGS
export update_hessian!, update_hessian_basis!
export ExactPenaltyCost, ExactPenaltyGrad, AugmentedLagrangianCost, AugmentedLagrangianGrad
#
# Stepsize
export Stepsize
export ArmijoLinesearch,
    ConstantStepsize, DecreasingStepsize, Linesearch, NonmonotoneLinesearch
export get_stepsize, get_initial_stepsize, get_last_stepsize
#
# Stopping Criteria
export StoppingCriterion, StoppingCriterionSet
export StopAfter,
    StopAfterIteration,
    StopWhenResidualIsReducedByFactorOrPower,
    StopWhenAll,
    StopWhenAny,
    StopWhenBundleLess,
    StopWhenChangeLess,
    StopWhenCostLess,
    StopWhenCostNan,
    StopWhenCurvatureIsNegative,
    StopWhenGradientNormLess,
    StopWhenIterNan,
    StopWhenSubgradientNormLess,
    StopWhenModelIncreased,
    StopWhenPopulationConcentrated,
    StopWhenSmallerOrEqual,
    StopWhenStepsizeLess,
    StopWhenTrustRegionIsExceeded
export get_active_stopping_criteria, get_stopping_criteria, get_reason
export update_stopping_criterion!
#
# Data functions
export artificial_S1_signal, artificial_S1_slope_signal, artificialIn_SAR_image
export artificial_SPD_image, artificial_SPD_image2
export artificial_S2_whirl_image, artificial_S2_whirl_patch
export artificial_S2_rotation_image
export artificial_S2_whirl_patch, artificial_S2_lemniscate
export artificial_S2_composite_bezier_curve
#
# Exports
export asymptote_export_S2_signals, asymptote_export_S2_data, asymptote_export_SPD
export render_asymptote
#
# Coeffs & Helpers for differentials
#
# Adjoint differentials
export adjoint_differential_forward_logs, adjoint_differential_forward_logs!
export adjoint_differential_bezier_control, adjoint_differential_bezier_control!
#
# Differentials
export differential_forward_logs, differential_forward_logs!
export differential_bezier_control, differential_bezier_control!
#
# Functions
export costL2TV, costL2TVTV2, costL2TV2, costTV, costTV2, costIntrICTV12
export cost_L2_acceleration_bezier, cost_acceleration_bezier
export ExactPenaltyCost, ExactPenaltyGrad
export SmoothingTechnique, LinearQuadraticHuber, LogarithmicSumOfExponentials
# Gradients
export grad_TV,
    grad_TV!,
    grad_TV2,
    grad_TV2!,
    grad_intrinsic_infimal_convolution_TV12,
    forward_logs,
    forward_logs!,
    grad_distance,
    grad_distance!,
    grad_acceleration_bezier,
    grad_L2_acceleration_bezier
# Proximal maps
export prox_distance, prox_distance!
export prox_TV, prox_TV!
export prox_parallel_TV, prox_parallel_TV!
export prox_TV2, prox_TV2!
export project_collaborative_TV, project_collaborative_TV!
# Error measures
export meanSquaredError, meanAverageError
#
# Bézier
export BezierSegment,
    de_casteljau,
    get_bezier_degrees,
    get_bezier_degree,
    get_bezier_inner_points,
    get_bezier_junction_tangent_vectors,
    get_bezier_junctions,
    get_bezier_points,
    get_bezier_segments
#
# Debugs
export DebugSolverState, DebugAction, DebugGroup, DebugEntry, DebugEntryChange, DebugEvery
export DebugChange,
    DebugGradientChange, DebugIterate, DebugIteration, DebugDivider, DebugTime
export DebugCost, DebugStoppingCriterion, DebugFactory, DebugActionFactory
export DebugGradient, DebugGradientNorm, DebugStepsize
export DebugPrimalBaseChange, DebugPrimalBaseIterate, DebugPrimalChange, DebugPrimalIterate
export DebugDualBaseChange, DebugDualBaseIterate, DebugDualChange, DebugDualIterate
export DebugDualResidual, DebugPrimalDualResidual, DebugPrimalResidual
export DebugProximalParameter, DebugWarnIfCostIncreases
export DebugGradient, DebugGradientNorm, DebugStepsize
export DebugWarnIfCostNotFinite, DebugWarnIfFieldNotFinite
#
# Records - and access functions
export get_record, get_record_state, get_record_action, has_record
export RecordAction
export RecordActionFactory, RecordFactory
export RecordGroup, RecordEvery
export RecordChange, RecordCost, RecordIterate, RecordIteration
export RecordEntry, RecordEntryChange, RecordTime
export RecordGradient, RecordGradientNorm, RecordStepsize
export RecordPrimalBaseChange,
    RecordPrimalBaseIterate, RecordPrimalChange, RecordPrimalIterate
export RecordDualBaseChange, RecordDualBaseIterate, RecordDualChange, RecordDualIterate
export RecordProximalParameter
#
# Helpers
export check_gradient, check_differential
end
