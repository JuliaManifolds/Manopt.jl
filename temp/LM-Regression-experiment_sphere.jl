#
#
# --- Regession on the sphere with outliers
using Distributions, GLMakie, Makie, ManifoldDiff, Manifolds, Manopt, NamedColors, Random, RecursiveArrayTools
ptc = NamedColors.load_paul_tol()

# Parameters
export_asy = true
show_plots = true
add_gaussian_noise = true
σ = π/12
# For outliers we use a fixed size and a random angle to disturb them into
r = π/4
N = 39 # on the range these are 0.05 apart for 39
outlier_indices = [3,5,7,35,37]

S = Manifolds.Sphere(2)
M = TangentBundle(S)
R(α) = [cos(α) sin(α); -sin(α) cos(α)]

# True data
p_true = [0.0, 1.0, 0.0]
X_true = 1.5 .* [1.0, 0.0, 1.0]
ts_true = collect(range(; start = -0.95, stop = 0.95, length=N))
qs_true = geodesic(S, p_true, X_true, ts_true)

if show_plots
    n = 30; u = range(0,stop=2*π,length=n); v = range(0,stop=π,length=n);
    sx = [cos(ui) * sin(vj) for ui in u, vj in v]
    sy = [sin(ui) * sin(vj) for ui in u, vj in v]
    sz = [cos(vj) for ui in u, vj in v]

    fig1 = Figure(size = (1400, 900), padding=0)
    ax1 = Axis3(fig1[1,1]; aspect =:data)
    hidedecorations!(ax1)
    hidespines!(ax1)
 	wireframe!(ax1, sx, sy, sz, color = ptc["paleblue"]; transparency=true, alpha = 0.2)
    geo_line = geodesic(S, p_true, X_true, range(-1.0, 1.0; length=1000))
    scatterlines!(
       ax1, Point3d.(geo_line); markersize=0, color=ptc["mutedteal"], linewidth=2,
    )
    scatter!(ax1, Point3d.([p_true,]); markersize=12, color=ptc["mutedteal"])
    scatter!(ax1, Point3d.(qs_true); markersize=8, color=ptc["mutedteal"])
    arrows3d!(
        ax1, Point3d.([p_true,]), Point3d.([X_true,]);
        color=ptc["mutedteal"], transparency=true, shaftradius=0.005, tiplength = 0.15, tipradius=0.025,
    )
end
Random.seed!(42)
data = [
    if i ∈ outlier_indices
        # sample random angle
        #α = rand(Uniform(0,2π))
        # all are outliers to the left or right
        α = π/2
        c = get_coordinates(S, q, parallel_transport_to(S, p_true, X_true, q), DefaultOrthonormalBasis())
        X_noise = get_vector(S, q, r/norm(c) .* R(α)*c, DefaultOrthonormalBasis())
        exp(S, q, X_noise)
    else
        exp(S, q, get_vector(S, q, add_gaussian_noise ? σ*randn(2) : zeros(2), DefaultOrthonormalBasis()))
    end for (i,q) in enumerate(qs_true)
]
show_plots && scatter!(ax1, Point3d.(data); markersize=12, color=ptc["mutedrose"])

# Cost (Vectorial) and its gradients (Jacobian) on the tangent bundle
# maybe as a struct / functor ?
function F(M, P; t=ts_true, d=data)
    S = base_manifold(M)
    p = P[M,:point]
    X = P[M, :vector]
    return [distance(S, geodesic(S, p, X, ti), di) for (ti,di) in zip(t,d)]
end
function JF(M, P; t=ts_true, d=data)
    S = base_manifold(M)
    p = P[M,:point]
    X = P[M, :vector]
    return [
        ArrayPartition(
            cost1_grad_p(S, p, X, ti, di),
            cost1_grad_X(S, p, X, ti, di),
        )
        for (ti,di) in zip(t,d)
    ]
end

function cost1(M::AbstractManifold, p, X, ti::Real, di)
    return distance(M, exp(M, p, ti * X), di)
end

function cost1_grad_p(M::AbstractManifold, p, X, ti::Real, di)
    z = exp(M, p, ti * X)
    gz = ManifoldDiff.grad_distance(M, di, z, 1)
    return ManifoldDiff.adjoint_differential_exp_basepoint(M, p, ti * X, gz)
end

function cost1_grad_X(M::AbstractManifold, p, X, ti::Real, di)
    z = exp(M, p, ti * X)
    gz = ManifoldDiff.grad_distance(M, di, z, 1)
    return ti * ManifoldDiff.adjoint_differential_exp_argument(M, p, ti * X, gz)
end

# This check requires plots which does not play so well with Makie
# Maybe also write a Makie extension?
#
# p0 = rand(S)
# X0 = rand(S; vector_at=p0)
# p1 = rand(S)
# Manopt.check_gradient(S,
#     (M, p) -> cost1(M, p, parallel_transport_to(M, p0, X0, p), 0.5, p1),
#     (M, p) -> cost1_grad_p(M, p, parallel_transport_to(M, p0, X0, p), 0.5, p1),
#     p0; plot=true
#     )

# Manopt.check_gradient(TangentSpace(S, p0),
#     (M, X) -> cost1(S, p0, X, 0.5, p1),
#     (M, X) -> cost1_grad_X(S, p0, X, 0.5, p1),
#     X0; plot=true
#     )

f = VectorGradientFunction(F, JF, N;
    evaluation = AllocatingEvaluation(),
    function_type = FunctionVectorialType(),
    jacobian_type = FunctionVectorialType(),
)

m = mean(S, data)
p0 = ArrayPartition(m, log(S, m, data[1]))

# Least Squares
P_star = LevenbergMarquardt(
    M, f, p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
    retraction_method = StabilizedRetraction(default_retraction_method(M)),
    debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop, 5],
)
p_star = P_star[M, :point]
X_star = P_star[M, :vector]

qs_star = geodesic(S, p_star, X_star, ts_true)

# Robust
P_ast = LevenbergMarquardt(
    M, f, p0;
    β = 8.0, η = 0.2, damping_term_min = 1.0e-5, ε = 1.0e-1, α_mode = :Strict,
    robustifier = 1e-7 ∘ HuberRobustifier(),
    retraction_method = StabilizedRetraction(default_retraction_method(M)),
    debug = [:Iteration, (:Cost, "f(x): %8.8e "), :damping_term, "\n", :Stop, 5],
)
p_ast = P_ast[M, :point]
X_ast = P_ast[M, :vector]

qs_ast = geodesic(S, p_ast, X_ast, ts_true)

if show_plots
    geo_line_mean = geodesic(S, p_star, X_star, range(-1.0, 1.0; length=1000))
    scatterlines!(
       ax1, Point3d.(geo_line_mean); markersize=0, color=ptc["mutedindigo"], linewidth=2,
    )
    scatter!(ax1, Point3d.([p_star,]); markersize=12, color=ptc["mutedindigo"])
    scatter!(ax1, Point3d.(qs_star); markersize=8, color=ptc["mutedindigo"])
    arrows3d!(
        ax1, Point3d.([p_star,]), Point3d.([X_star,]);
        color=ptc["mutedindigo"], transparency=true, shaftradius=0.005, tiplength = 0.15, tipradius=0.025,
    )

    geo_line_robust = geodesic(S, p_ast, X_ast, range(-1.0, 1.0; length=1000))
    scatterlines!(
       ax1, Point3d.(geo_line_robust); markersize=0, color=ptc["mutedgreen"], linewidth=2,
    )
    scatter!(ax1, Point3d.([p_ast,]); markersize=12, color=ptc["mutedgreen"])
    scatter!(ax1, Point3d.(qs_ast); markersize=8, color=ptc["mutedgreen"])
    arrows3d!(
        ax1, Point3d.([p_ast,]), Point3d.([X_ast,]);
        color=ptc["mutedgreen"], transparency=true, shaftradius=0.005, tiplength = 0.15, tipradius=0.025,
    )
end

@info "Error on sample points least squares: $(norm([distance(S, qi, qmi) for (qi, qmi) in zip(qs_true, qs_star)]))"
@info "Error on sample points robust: $(norm([distance(S, qi, qri) for (qi, qri) in zip(qs_true, qs_ast)]))"
fig1