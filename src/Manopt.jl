@doc """
🏔️ Manopt.jl: optimization on Manifolds in Julia.

* 📚 Documentation: [manoptjl.org](https://manoptjl.org)
* 📦 Repository: [github.com/JuliaManifolds/Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl)
* 💬 Discussions: [github.com/JuliaManifolds/Manopt.jl/discussions](https://github.com/JuliaManifolds/Manopt.jl/discussions)
* 🎯 Issues: [github.com/JuliaManifolds/Manopt.jl/issues](https://github.com/JuliaManifolds/Manopt.jl/issues)
"""
module Manopt

import Base: &, copy, getindex, identity, length, setindex!, show, |
import LinearAlgebra: reflect!
import ManifoldsBase: embed!, plot_slope, prepare_check_result, find_best_slope_window
import ManifoldsBase: base_manifold, base_point, get_basis
import ManifoldsBase: project, project!
import LinearAlgebra: cross
using ColorSchemes
using ColorTypes
using Colors
using DataStructures: CircularBuffer, capacity, length, push!, size, isfull
using Dates: Millisecond, Nanosecond, Period, canonicalize, value
using Glossaries
using LinearAlgebra:
    cond,
    Adjoint,
    Diagonal,
    I,
    Eigen,
    LinearAlgebra,
    PosDefException,
    eigen,
    eigen!,
    eigvals,
    ldiv!,
    tril,
    Symmetric,
    dot,
    cholesky,
    eigmin,
    opnorm,
    mul!
using ManifoldDiff:
    adjoint_differential_log_argument,
    adjoint_differential_log_argument!,
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
    jacobi_field!,
    riemannian_gradient,
    riemannian_gradient!,
    riemannian_Hessian,
    riemannian_Hessian!
using ManifoldsBase
using ManifoldsBase:
    AbstractBasis,
    AbstractDecoratorManifold,
    AbstractInverseRetractionMethod,
    AbstractManifold,
    AbstractPowerManifold,
    AbstractPowerRepresentation,
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
    ProductManifold,
    ProjectionTransport,
    QRRetraction,
    TangentSpace,
    ^,
    _read,
    _write,
    allocate,
    allocate_result,
    allocate_result_type,
    base_manifold,
    copy,
    copyto!,
    default_inverse_retraction_method,
    default_retraction_method,
    default_vector_transport_method,
    distance,
    embed,
    embed_project,
    embed_project!,
    exp,
    exp!,
    geodesic,
    get_basis,
    get_component,
    get_coordinates,
    get_coordinates!,
    get_embedding,
    get_iterator,
    get_vector,
    get_vector!,
    get_vectors,
    has_components,
    injectivity_radius,
    inner,
    inverse_retract,
    inverse_retract!,
    is_flat,
    is_point,
    is_vector,
    log,
    log!,
    manifold_dimension,
    mid_point,
    mid_point!,
    norm,
    number_eltype,
    number_of_coordinates,
    power_dimensions,
    project,
    project!,
    rand!,
    riemann_tensor,
    riemann_tensor!,
    representation_size,
    requires_caching,
    retract,
    retract!,
    sectional_curvature,
    set_component!,
    shortest_geodesic,
    shortest_geodesic!,
    submanifold_components,
    vector_transport_to,
    vector_transport_to!,
    zero_vector,
    zero_vector!,
    ×,
    ℂ,
    ℝ
using Markdown
using Preferences:
    @load_preference, @set_preferences!, @has_preference, @delete_preferences!
using Printf
using Random: AbstractRNG, default_rng, shuffle!, rand, randn!, randperm
using SparseArrays
using Statistics

include("documentation_glossary.jl")

"""
    Rn(args; kwargs...)
    Rn(s::Symbol=:Manifolds, args; kwargs...)

A small internal helper function to choose a Euclidean space.
By default, this uses the [`DefaultManifold`](@extref ManifoldsBase.DefaultManifold) unless you load
a more advanced Euclidean space like [`Euclidean`](@extref Manifolds.Euclidean)
from [`Manifolds.jl`](@extref Manifolds.Manifolds)
"""
Rn(args...; kwargs...) = Rn(Val(Rn_default()), args...; kwargs...)

@doc """
    Rn_default()

Specify a default value to dispatch [`Rn`](@ref) on.
This default is set to `Manifolds`, indicating, that when this package is loaded,
it is the preferred package to ask for a vector space space.

The default within `Manopt.jl` is to use the [`DefaultManifold`](@extref ManifoldsBase.DefaultManifold) from `ManifoldsBase.jl`.
If you load `Manifolds.jl` this switches to using [`Euclidean`](@extref Manifolds.Euclidean).
"""
Rn_default() = :Manifolds
Rn(::Val{T}, args...; kwargs...) where {T} = DefaultManifold(args...; kwargs...)

include("plans/plan.jl")
# solvers general framework
include("solvers/solver.jl")
# specific solvers
include("solvers/adaptive_regularization_with_cubics.jl")
include("solvers/alternating_gradient_descent.jl")
include("solvers/augmented_Lagrangian_method.jl")
include("solvers/convex_bundle_method.jl")
include("solvers/ChambollePock.jl")
include("solvers/cma_es.jl")
include("solvers/conjugate_gradient_descent.jl")
include("solvers/conjugate_residual.jl")
include("solvers/cyclic_proximal_point.jl")
include("solvers/difference_of_convex_algorithm.jl")
include("solvers/difference_of_convex_proximal_point.jl")
include("solvers/DouglasRachford.jl")
include("solvers/exact_penalty_method.jl")
include("solvers/projected_gradient_method.jl")
include("solvers/Lanczos.jl")
include("solvers/NelderMead.jl")
include("solvers/FrankWolfe.jl")
include("solvers/gradient_descent.jl")
include("solvers/interior_point_Newton.jl")
include("solvers/LevenbergMarquardt.jl")
include("solvers/mesh_adaptive_direct_search.jl")
include("solvers/particle_swarm.jl")
include("solvers/primal_dual_semismooth_Newton.jl")
include("solvers/proximal_bundle_method.jl")
include("solvers/proximal_gradient_method.jl")
include("solvers/proximal_point.jl")
include("solvers/quasi_Newton.jl")
include("solvers/truncated_conjugate_gradient_descent.jl")
include("solvers/trust_regions.jl")
include("solvers/stochastic_gradient_descent.jl")
include("solvers/subgradient.jl")
include("solvers/vectorbundle_newton.jl")
include("solvers/debug_solver.jl")
include("solvers/record_solver.jl")

include("helpers/checks.jl")
include("helpers/jacobian_block.jl")
include("helpers/exports/Asymptote.jl")
include("helpers/LineSearchesTypes.jl")
include("helpers//test.jl")

include("deprecated.jl")

function JuMP_Optimizer end

function __init__()
    #
    # Error Hints
    #
    @static if isdefined(Base.Experimental, :register_error_hint) # COV_EXCL_LINE
        Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
            if exc.f === convex_bundle_method_subsolver
                print(
                    io,
                    "\nThe `convex_bundle_method_subsolver` has to be implemented. A default is available currently when loading QuadraticModels.jl and RipQP.jl. That is\n",
                )
                printstyled(io, "`using QuadraticModels, RipQP`"; color = :cyan)
            end
            if exc.f === proximal_bundle_method_subsolver
                print(
                    io,
                    "\nThe `proximal_bundle_method_subsolver` has to be implemented. A default is available currently when loading QuadraticModels.jl and RipQP.jl. That is\n",
                )
                printstyled(io, "`using QuadraticModels, RipQP`"; color = :cyan)
            end
            if exc.f === Manopt.JuMP_Optimizer
                print(
                    io,
                    """

                    The `Manopt.JuMP_Optimizer` is not yet properly initialized.
                    It requires the package `JuMP.jl`, so please load it e.g. via
                    """,
                )
                printstyled(io, "`using JuMP`"; color = :cyan)
            end
        end
    end
    return nothing
end
#
# General
export ℝ, ℂ, &, |, ×, ≟, ⩼, ⩻
export mid_point, mid_point!, reflect, reflect!
#
# Problems
export AbstractManoptProblem
export DefaultManoptProblem
export TwoManifoldProblem, ConstrainedManoptProblem, VectorBundleManoptProblem
#
# Objectives
export AbstractDecoratedManifoldObjective,
    AbstractManifoldFirstOrderObjective,
    AbstractManifoldCostObjective,
    AbstractManifoldObjective,
    AbstractManifoldSubObjective,
    AbstractPrimalDualManifoldObjective,
    ConstrainedManifoldObjective,
    ManifoldConstrainedSetObjective,
    EmbeddedManifoldObjective,
    ScaledManifoldObjective,
    ManifoldCountObjective,
    NonlinearLeastSquaresObjective,
    ManifoldAlternatingGradientObjective,
    ManifoldCostGradientObjective,
    ManifoldCostObjective,
    ManifoldDifferenceOfConvexObjective,
    ManifoldDifferenceOfConvexProximalObjective,
    ManifoldFirstOrderObjective,
    ManifoldGradientObjective,
    ManifoldHessianObjective,
    ManifoldProximalGradientObjective,
    ManifoldProximalMapObjective,
    ManifoldStochasticGradientObjective,
    ManifoldSubgradientObjective,
    NonlinearLeastSquaresObjective,
    PrimalDualManifoldObjective,
    PrimalDualManifoldSemismoothNewtonObjective,
    SimpleManifoldCachedObjective,
    ManifoldCachedObjective
# Functions
export AbstractVectorFunction,
    AbstractVectorGradientFunction, VectorGradientFunction, VectorHessianFunction
# Robustifiers
export AbstractRobustifierFunction, SoftL1Robustifier, AbstractRobustifierFunction,
    CauchyRobustifier, TolerantRobustifier, TukeyRobustifier, ComposedRobustifierFunction,
    ArctanRobustifier, ScaledRobustifierFunction, RobustifierFunction, IdentityRobustifier,
    HuberRobustifier, ComponentwiseRobustifierFunction
#
# Evaluation & Vectorial Types
export AbstractEvaluationType, AllocatingEvaluation, InplaceEvaluation, evaluation_type
export AbstractVectorialType
export CoefficientVectorialType, ComponentVectorialType, FunctionVectorialType
#
# AbstractManoptSolverState
export AbstractGradientSolverState,
    AbstractHessianSolverState,
    AbstractManoptSolverState,
    AbstractPrimalDualSolverState,
    AdaptiveRegularizationState,
    AlternatingGradientDescentState,
    AugmentedLagrangianMethodState,
    ConvexBundleMethodState,
    ChambollePockState,
    ConjugateGradientDescentState,
    ConjugateResidualState,
    CoordinatesNormalSystemState,
    CyclicProximalPointState,
    DifferenceOfConvexState,
    DifferenceOfConvexProximalState,
    DouglasRachfordState,
    ExactPenaltyMethodState,
    FrankWolfeState,
    GradientDescentState,
    InteriorPointNewtonState,
    LanczosState,
    LevenbergMarquardtState,
    MeshAdaptiveDirectSearchState,
    NelderMeadState,
    ParticleSwarmState,
    PrimalDualSemismoothNewtonState,
    ProjectedGradientMethodState,
    ProximalBundleMethodState,
    ProximalGradientMethodState,
    RecordSolverState,
    StepsizeState,
    StochasticGradientDescentState,
    SubGradientMethodState,
    TruncatedConjugateGradientState,
    TrustRegionsState,
    VectorBundleNewtonState

# Objectives and Costs
export NelderMeadSimplex
export AlternatingGradient
#
# access functions and helpers for `AbstractManoptSolverState`
export default_stepsize
export get_cost, get_gradient, get_gradient!, get_cost_smooth
export get_subgradient, get_subgradient!
export get_subtrahend_gradient!, get_subtrahend_gradient
export get_proximal_map, get_proximal_map!
export get_state,
    get_initial_stepsize,
    get_iterate,
    get_adjoint_jacobian,
    get_adjoint_jacobian!,
    get_jacobian,
    get_jacobian!,
    get_gradients,
    get_gradients!,
    get_manifold,
    get_preconditioner,
    get_preconditioner!,
    get_primal_prox,
    get_primal_prox!,
    get_projected_point,
    get_projected_point!,
    get_differential_primal_prox,
    get_differential_primal_prox!,
    get_dual_prox,
    get_dual_prox!,
    get_differential_dual_prox,
    get_differential_dual_prox!,
    set_gradient!,
    set_iterate!,
    get_residuals,
    get_residuals!,
    has_converged,
    linearized_forward_operator,
    linearized_forward_operator!,
    adjoint_linearized_operator,
    adjoint_linearized_operator!,
    forward_operator,
    forward_operator!,
    get_objective,
    get_unconstrained_objective
export get_hessian, get_hessian!
export get_differential
export ApproxHessianFiniteDifference
export is_state_decorator, dispatch_state_decorator
export primal_residual, dual_residual
export equality_constraints_length,
    get_constraints,
    get_inequality_constraint,
    get_equality_constraint,
    get_grad_inequality_constraint,
    get_grad_inequality_constraint!,
    get_grad_equality_constraint,
    get_grad_equality_constraint!,
    get_hess_inequality_constraint,
    get_hess_inequality_constraint!,
    get_hess_equality_constraint,
    get_hess_equality_constraint!,
    get_robustifier_values,
    inequality_constraints_length,
    is_feasible
# Subproblem cost/grad
export AugmentedLagrangianCost, AugmentedLagrangianGrad, ExactPenaltyCost, ExactPenaltyGrad
export KKTVectorField, KKTVectorFieldJacobian, KKTVectorFieldAdjointJacobian
export KKTVectorFieldNormSq, KKTVectorFieldNormSqGradient
export LagrangianCost, LagrangianGradient, LagrangianHessian
export ProximalDCCost, ProximalDCGrad, LinearizedDCCost, LinearizedDCGrad
export FrankWolfeCost, FrankWolfeGradient
export LevenbergMarquardtLinearSurrogateObjective
export TrustRegionModelObjective
export CondensedKKTVectorField, CondensedKKTVectorFieldJacobian
export SymmetricLinearSystemObjective
export ProximalGradientNonsmoothCost, ProximalGradientNonsmoothSubgradient

export QuasiNewtonState, QuasiNewtonLimitedMemoryDirectionUpdate
export QuasiNewtonMatrixDirectionUpdate
export QuasiNewtonPreconditioner
export QuasiNewtonCautiousDirectionUpdate,
    BFGS, InverseBFGS, DFP, InverseDFP, SR1, InverseSR1
export InverseBroyden, Broyden
export AbstractQuasiNewtonDirectionUpdate, AbstractQuasiNewtonUpdateRule
export WolfePowellLinesearch, WolfePowellBinaryLinesearch
export AbstractStateAction, StoreStateAction
export has_storage, get_storage, update_storage!
export objective_cache_factory
export AbstractMeshPollFunction, LowerTriangularAdaptivePoll
export AbstractMeshSearchFunction, DefaultMeshAdaptiveDirectSearch
#
# Direction Update Rules
export DirectionUpdateRule
export Gradient, StochasticGradient
export AverageGradient, MomentumGradient, Nesterov, PreconditionedDirection
export SteepestDescentCoefficient,
    HestenesStiefelCoefficient,
    FletcherReevesCoefficient,
    PolakRibiereCoefficient,
    ConjugateDescentCoefficient,
    LiuStoreyCoefficient,
    DaiYuanCoefficient,
    HagerZhangCoefficient,
    ConjugateGradientBealeRestart,
    HybridCoefficient
#
# Restart Conditions
export AbstractRestartCondition
export NeverRestart,
    RestartOnNonDescent,
    RestartOnNonSufficientDescent
#
#
# Solvers
export adaptive_regularization_with_cubics,
    accepted_keywords,
    adaptive_regularization_with_cubics!,
    alternating_gradient_descent,
    alternating_gradient_descent!,
    augmented_Lagrangian_method,
    augmented_Lagrangian_method!,
    convex_bundle_method,
    convex_bundle_method!,
    convex_bundle_method_subsolver,
    convex_bundle_method_subsolver!,
    ChambollePock,
    ChambollePock!,
    cma_es,
    cma_es!,
    conjugate_gradient_descent,
    conjugate_gradient_descent!,
    conjugate_residual,
    conjugate_residual!,
    cyclic_proximal_point,
    cyclic_proximal_point!,
    difference_of_convex_algorithm,
    difference_of_convex_algorithm!,
    difference_of_convex_proximal_point,
    difference_of_convex_proximal_point!,
    DouglasRachford,
    DouglasRachford!,
    exact_penalty_method,
    exact_penalty_method!,
    Frank_Wolfe_method,
    Frank_Wolfe_method!,
    gradient_descent,
    gradient_descent!,
    interior_point_Newton,
    interior_point_Newton!,
    LevenbergMarquardt,
    LevenbergMarquardt!,
    mesh_adaptive_direct_search,
    mesh_adaptive_direct_search!,
    NelderMead,
    NelderMead!,
    particle_swarm,
    particle_swarm!,
    primal_dual_semismooth_Newton,
    projected_gradient_method,
    projected_gradient_method!,
    proximal_bundle_method,
    proximal_bundle_method!,
    proximal_gradient_method,
    proximal_gradient_method!,
    proximal_point,
    proximal_point!,
    quasi_Newton,
    quasi_Newton!,
    stochastic_gradient_descent,
    stochastic_gradient_descent!,
    subgradient_method,
    subgradient_method!,
    truncated_conjugate_gradient_descent,
    truncated_conjugate_gradient_descent!,
    trust_regions,
    trust_regions!,
    vectorbundle_newton,
    vectorbundle_newton!
#
# Solver helpers
export decorate_state!, decorate_objective!
export initialize_solver!, step_solver!, get_solver_result, stop_solver!
export solve!
export ApproxHessianFiniteDifference, ApproxHessianSymmetricRankOne, ApproxHessianBFGS
export update_hessian!, update_hessian_basis!
export ExactPenaltyCost, ExactPenaltyGrad, AugmentedLagrangianCost, AugmentedLagrangianGrad
export AdaptiveRegularizationWithCubicsModelObjective
export ExactPenaltyCost, ExactPenaltyGrad
export SmoothingTechnique, LinearQuadraticHuber, LogarithmicSumOfExponentials
#
# Stepsize
export Stepsize
export AdaptiveWNGradient, AffineCovariantStepsize, ConstantLength, DecreasingLength,
    Polyak, DistanceOverGradients, DistanceOverGradientsStepsize
export ProximalGradientMethodBacktracking
export ArmijoLinesearch, Linesearch, NonmonotoneLinesearch, CubicBracketingLinesearch
export get_stepsize, get_initial_stepsize, get_last_stepsize
export InteriorPointCentralityCondition
export DomainBackTracking, DomainBackTrackingStepsize, NullStepBackTrackingStepsize
export ProximalGradientMethodBacktracking
#
# Stopping Criteria
export StoppingCriterion, StoppingCriterionSet
export StopAfter,
    StopAfterIteration,
    StopWhenResidualIsReducedByFactorOrPower,
    StopWhenAll,
    StopWhenAllLanczosVectorsUsed,
    StopWhenAny,
    StopWhenBestCostInGenerationConstant,
    StopWhenChangeLess,
    StopWhenCostLess,
    StopWhenCostChangeLess,
    StopWhenCostNaN,
    StopWhenCovarianceIllConditioned,
    StopWhenCurvatureIsNegative,
    StopWhenCriterionWithIterationCondition,
    StopWhenEntryChangeLess,
    StopWhenEvolutionStagnates,
    StopWhenGradientChangeLess,
    StopWhenGradientMappingNormLess,
    StopWhenGradientNormLess,
    StopWhenFirstOrderProgress,
    StopWhenIterateNaN,
    StopWhenKKTResidualLess,
    StopWhenLagrangeMultiplierLess,
    StopWhenModelIncreased,
    StopWhenPollSizeLess,
    StopWhenPopulationCostConcentrated,
    StopWhenPopulationConcentrated,
    StopWhenPopulationDiverges,
    StopWhenPopulationStronglyConcentrated,
    StopWhenProjectedGradientStationary,
    StopWhenRelativeResidualLess,
    StopWhenRepeated,
    StopWhenSmallerOrEqual,
    StopWhenStepsizeLess,
    StopWhenSubgradientNormLess,
    StopWhenSwarmVelocityLess,
    StopWhenTrustRegionIsExceeded
export get_active_stopping_criteria,
    get_stopping_criteria, get_reason, get_stopping_criterion
#
# Exports
export asymptote_export_S2_signals, asymptote_export_S2_data, asymptote_export_SPD
export render_asymptote
#
# Debugs
export DebugSolverState, DebugAction, DebugGroup, DebugEntry, DebugEntryChange, DebugEvery
export DebugChange, DebugGradientChange
export DebugIterate, DebugIteration, DebugDivider, DebugTime
export DebugFeasibility
export DebugCost, DebugStoppingCriterion
export DebugGradient, DebugGradientNorm, DebugStepsize
export DebugPrimalBaseChange, DebugPrimalBaseIterate, DebugPrimalChange, DebugPrimalIterate
export DebugDualBaseChange, DebugDualBaseIterate, DebugDualChange, DebugDualIterate
export DebugDualResidual, DebugPrimalDualResidual, DebugPrimalResidual
export DebugProximalParameter, DebugWarnIfCostIncreases
export DebugGradient, DebugGradientNorm, DebugStepsize
export DebugWhenActive, DebugWarnIfFieldNotFinite, DebugIfEntry
export DebugWarnIfCostNotFinite, DebugWarnIfFieldNotFinite
export DebugWarnIfLagrangeMultiplierIncreases, DebugWarnIfStepsizeCollapsed
export DebugWarnIfGradientNormTooLarge, DebugMessages
#
# Records - and access functions
export get_record, get_record_state, get_record_action, has_record, getindex
export RecordAction
export RecordGroup, RecordEvery
export RecordChange, RecordCost, RecordIterate, RecordIteration
export RecordEntry, RecordEntryChange, RecordTime
export RecordGradient, RecordGradientNorm, RecordStepsize
export RecordSubsolver, RecordWhenActive, RecordStoppingReason
export RecordPrimalBaseChange,
    RecordPrimalBaseIterate, RecordPrimalChange, RecordPrimalIterate
export RecordStoppingReason, RecordWhenActive, RecordSubsolver
export RecordDualBaseChange, RecordDualBaseIterate, RecordDualChange, RecordDualIterate
export RecordProximalParameter
#
# Count
export get_count, reset_counters!
#
# Helpers
export check_gradient, check_differential, check_Hessian
export JacobianBlock
end
