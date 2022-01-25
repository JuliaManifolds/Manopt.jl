"""
`Manopt.jl` – Optimization on Manifolds in Julia.
"""
module Manopt
using Colors
using ColorSchemes
using ColorTypes
using Markdown
using LinearAlgebra: I, Diagonal, eigvals, eigen, tril
using Dates: Period, Nanosecond, value
using Requires
using Random: shuffle!
using DataStructures: CircularBuffer, capacity, length, size, push!
using StaticArrays
import Base: copy, identity, &, |
import ManifoldsBase:
    ℝ,
    ℂ,
    ×,
    ^,
    _read,
    _write,
    AbstractBasis,
    AbstractPowerManifold,
    AbstractVectorTransportMethod,
    AbstractRetractionMethod,
    AbstractInverseRetractionMethod,
    CachedBasis,
    DefaultOrthonormalBasis,
    ExponentialRetraction,
    LogarithmicInverseRetraction,
    ParallelTransport,
    PowerManifold,
    AbstractManifold,
    allocate,
    allocate_result,
    allocate_result_type,
    copyto!,
    default_inverse_retraction_method,
    default_retraction_method,
    default_vector_transport_method,
    distance,
    exp,
    exp!,
    log,
    log!,
    injectivity_radius,
    inner,
    geodesic,
    get_basis,
    get_component,
    get_coordinates,
    get_vector,
    get_vectors,
    get_iterator,
    getindex,
    manifold_dimension,
    mid_point,
    mid_point!,
    NestedPowerRepresentation,
    norm,
    number_eltype,
    power_dimensions,
    project,
    project!,
    retract,
    retract!,
    inverse_retract,
    inverse_retract!,
    shortest_geodesic,
    vector_transport_to,
    vector_transport_to!,
    zero_vector,
    zero_vector!,
    DiagonalizingOrthonormalBasis,
    representation_size,
    setindex!,
    set_component!

include("plans/plan.jl")
# Functions
include("functions/bezier_curves.jl")
include("functions/adjoint_differentials.jl")
include("functions/costs.jl")
include("functions/differentials.jl")
include("functions/gradients.jl")
include("functions/Jacobi_fields.jl")
include("functions/proximal_maps.jl")
# solvers general framework
include("solvers/solver.jl")
# specific solvers
include("solvers/ChambollePock.jl")
include("solvers/conjugate_gradient_descent.jl")
include("solvers/cyclic_proximal_point.jl")
include("solvers/DouglasRachford.jl")
include("solvers/NelderMead.jl")
include("solvers/gradient_descent.jl")
include("solvers/particle_swarm.jl")
include("solvers/quasi_Newton.jl")
include("solvers/truncated_conjugate_gradient_descent.jl")
include("solvers/trust_regions.jl")
include("solvers/stochastic_gradient_descent.jl")
include("solvers/subgradient.jl")
include("solvers/debug_solver.jl")
include("solvers/record_solver.jl")
include("helpers/errorMeasures.jl")
include("helpers/exports/Asymptote.jl")
include("data/artificialDataFunctions.jl")

function __init__()
    @require Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e" begin
        using .Manifolds:
            AbstractGroupManifold,
            Circle,
            Euclidean,
            Grassmann,
            Hyperbolic,
            PositiveNumbers,
            ProductManifold,
            Rotations,
            SymmetricPositiveDefinite,
            Stiefel,
            Sphere,
            TangentBundle,
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
        include("helpers/random.jl")
        # adaptions for Nonmutating manifolds
        const NONMUTATINGMANIFOLDS = Union{Circle,PositiveNumbers,Euclidean{Tuple{}}}
        include("functions/manifold_functions.jl")
        include("functions/nonmutating_manifolds_functions.jl")
        include("plans/nonmutating_manifolds_plans.jl")
        include("plans/alternating_gradient_plan.jl")
        include("solvers/alternating_gradient_descent.jl")
        export random_point, random_tangent, mid_point, mid_point!, reflect, reflect!
        export AlternatingGradientDescentOptions, AlternatingGradientProblem
        export AlternatingGradient
        export alternating_gradient_descent, alternating_gradient_descent!
    end
    return nothing
end
#
# General
export ℝ, ℂ, &, |
#
# Problems
export Problem,
    ProximalProblem,
    CostProblem,
    SubGradientProblem,
    GradientProblem,
    HessianProblem,
    PrimalDualProblem,
    StochasticGradientProblem,
    AbstractEvaluationType,
    AllocatingEvaluation,
    MutatingEvaluation
#
# Options
export Options,
    AbstractGradientOptions,
    ChambollePockOptions,
    ConjugateGradientDescentOptions,
    CyclicProximalPointOptions,
    DouglasRachfordOptions,
    GradientDescentOptions,
    AbstractHessianOptions,
    NelderMeadOptions,
    ParticleSwarmOptions,
    PrimalDualOptions,
    RecordOptions,
    StochasticGradientDescentOptions,
    SubGradientMethodOptions,
    TruncatedConjugateGradientOptions,
    TrustRegionsOptions
#
# Accessors and helpers for Options
export linesearch_backtrack
export get_cost,
    get_gradient,
    get_gradient!,
    get_subgradient,
    get_subgradient!,
    get_proximal_map,
    get_proximal_map!,
    get_options,
    get_initial_stepsize,
    get_gradients,
    get_gradients!,
    get_primal_prox,
    get_primal_prox!,
    get_dual_prox,
    get_dual_prox!,
    linearized_forward_operator,
    linearized_forward_operator!,
    adjoint_linearized_operator,
    adjoint_linearized_operator!,
    forward_operator,
    forward_operator!
export get_hessian, get_hessian!, ApproxHessianFiniteDifference
export is_options_decorator, dispatch_options_decorator
export primal_residual, dual_residual

export QuasiNewtonOptions, QuasiNewtonLimitedMemoryDirectionUpdate
export QuasiNewtonCautiousDirectionUpdate,
    BFGS, InverseBFGS, DFP, InverseDFP, SR1, InverseSR1
export InverseBroyden, Broyden
export AbstractQuasiNewtonDirectionUpdate, AbstractQuasiNewtonUpdateRule
export WolfePowellLineseach,
    StrongWolfePowellLineseach,
    operator_to_matrix,
    square_matrix_vector_product,
    WolfePowellBinaryLinesearch

export ConjugateGradientDescentOptions,
    GradientDescentOptions,
    AbstractHessianOptions,
    SubGradientMethodOptions,
    NelderMeadOptions,
    TruncatedConjugateGradientOptions,
    TrustRegionsOptions,
    ParticleSwarmOptions
export AbstractOptionsAction, StoreOptionsAction
export has_storage, get_storage, update_storage!

#
# Direction Update Rules
export DirectionUpdateRule,
    IdentityUpdateRule, StochasticGradient, AverageGradient, MomentumGradient, Nesterov
export DirectionUpdateRule,
    SteepestDirectionUpdateRule,
    HeestenesStiefelCoefficient,
    FletcherReevesCoefficient,
    PolakRibiereCoefficient,
    ConjugateDescentCoefficient,
    LiuStoreyCoefficient,
    DaiYuanCoefficient,
    HagerZhangCoefficient
#
# Solvers
export ChambollePock,
    ChambollePock!,
    conjugate_gradient_descent,
    conjugate_gradient_descent!,
    cyclic_proximal_point,
    cyclic_proximal_point!,
    DouglasRachford,
    DouglasRachford!,
    gradient_descent,
    gradient_descent!,
    NelderMead,
    NelderMead!,
    particle_swarm,
    particle_swarm!,
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
export decorate_options
export initialize_solver!, step_solver!, get_solver_result, stop_solver!
export solve
export ApproxHessianFiniteDifference
#
# Stepsize
export ConstantStepsize, DecreasingStepsize
export Linesearch, ArmijoLinesearch, NonmonotoneLinesearch
export get_stepsize, get_initial_stepsize, get_last_stepsize
#
# Stopping Criteria
export StopIfResidualIsReducedByFactor,
    StopIfResidualIsReducedByPower,
    StopWhenCurvatureIsNegative,
    StopWhenTrustRegionIsExceeded,
    StopWhenModelIncreased
export StopAfterIteration, StopWhenChangeLess, StopWhenGradientNormLess, StopWhenCostLess
export StopAfter, StopWhenAll, StopWhenAny
export get_active_stopping_criteria, get_stopping_criteria, get_reason
export are_these_stopping_critera_active
export StoppingCriterion, StoppingCriterionSet, Stepsize
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
export βdifferential_geodesic_startpoint, βdifferential_exp_basepoint
export βdifferential_exp_argument, βdifferential_log_basepoint, βdifferential_log_argument
export jacobi_field, adjoint_Jacobi_field
#
# Adjoint differentials
export adjoint_differential_geodesic_startpoint, adjoint_differential_geodesic_startpoint!
export adjoint_differential_geodesic_endpoint, adjoint_differential_geodesic_endpoint!
export adjoint_differential_exp_basepoint, adjoint_differential_exp_basepoint!
export adjoint_differential_exp_argument, adjoint_differential_exp_argument!
export adjoint_differential_log_basepoint, adjoint_differential_log_basepoint!
export adjoint_differential_log_argument, adjoint_differential_log_argument!
export adjoint_differential_forward_logs, adjoint_differential_forward_logs!
export adjoint_differential_bezier_control, adjoint_differential_bezier_control!
#
# Differentials
export differential_geodesic_startpoint, differential_geodesic_startpoint!
export differential_geodesic_endpoint, differential_geodesic_endpoint!
export differential_exp_basepoint, differential_exp_basepoint!
export differential_exp_argument, differential_exp_argument!
export differential_log_basepoint, differential_log_basepoint!
export differential_log_argument, differential_log_argument!
export differential_forward_logs, differential_forward_logs!
export differential_bezier_control, differential_bezier_control!
#
# Functions
export costL2TV, costL2TVTV2, costL2TV2, costTV, costTV2, costIntrICTV12
export cost_L2_acceleration_bezier, cost_acceleration_bezier
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
export DebugOptions, DebugAction, DebugGroup, DebugEntry, DebugEntryChange, DebugEvery
export DebugChange, DebugIterate, DebugIteration, DebugDivider
export DebugCost, DebugStoppingCriterion, DebugFactory, DebugActionFactory
export DebugGradient, DebugGradientNorm, DebugStepsize
export DebugPrimalBaseChange, DebugPrimalBaseIterate, DebugPrimalChange, DebugPrimalIterate
export DebugDualBaseChange, DebugDualBaseIterate, DebugDualChange, DebugDualIterate
export DebugDualResidual, DebugPrimalDualResidual, DebugPrimalResidual
export DebugProximalParameter
export DebugGradient, DebugGradientNorm, DebugStepsize
#
# Records - and access functions
export get_record, get_record_options, get_record_action, has_record
export RecordAction
export RecordActionFactory, RecordFactory
export RecordGroup, RecordEvery
export RecordChange, RecordCost, RecordIterate, RecordIteration
export RecordEntry, RecordEntryChange
export RecordGradient, RecordGradientNorm, RecordStepsize
export RecordPrimalBaseChange,
    RecordPrimalBaseIterate, RecordPrimalChange, RecordPrimalIterate
export RecordDualBaseChange, RecordDualBaseIterate, RecordDualChange, RecordDualIterate
export RecordProximalParameter
end
