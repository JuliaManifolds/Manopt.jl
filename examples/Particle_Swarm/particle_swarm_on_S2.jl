using Manopt, Manifolds, Makie, Colors, LinearAlgebra, ColorSchemes, Random
Random.seed!(42)
#
# Helpers and Function
rot_x(φ) = [1 0 0; 0 cos(φ) -sin(φ); 0 sin(φ) cos(φ)]
rot_y(φ) = [cos(φ) 0 -sin(φ); 0 1 0; sin(φ) 0 cos(φ)]
rot_z(φ) = [cos(φ) -sin(φ) 0; sin(φ) cos(φ) 0; 0 0 1]
spherical_to_euclidean(θ, φ, r = 1) = [r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)]
function euclidean_to_spherical(x, y, z)
    return [acos(z / sqrt(x^2 + y^2 + z^2)), atan(y, x), sqrt(x^2 + y^2 + z^2)]
end

# A = Diagonal([2.0, 1.0, 0.5])
# B = rot_z(pi/4)*Diagonal([4.0, 1.0, 2.0])rot_z(-π/4)
# C = rot_x(π/8)*rot_y(-π/4)*Diagonal([0.5, 2.0, 3.0])*rot_y(π/4)*rot_x(-π/8)
# D = A+B-C
# F(x) = x'*D*x
bob(x, x0, r = 1) = (norm(x - x0) <= r) ? 1 - norm(x - x0) / r : 0
p1 = [0.0, 0.0, 1.0]
function F(x)
    s = euclidean_to_spherical(x[1], x[2], x[3])
    θ = s[1]
    φ = mod(s[2], 2π)
    v = sin(5θ + 3φ) + sin(3θ - 5φ) - 0.5 * (bob(x, p1, π / 2)^2 + bob(x, -p1, π / 2)^2)
    return v
end
M = Manifolds.Sphere(2)

#
# Plot Settings
n_lon = 801
n_lat = 401
lat = range(0, π; length = n_lat)
lon = range(-π, π; length = n_lon)
mesh_x = [spherical_to_euclidean(θ, φ)[1] for θ in lat, φ in lon]
mesh_y = [spherical_to_euclidean(θ, φ)[2] for θ in lat, φ in lon]
mesh_z = [spherical_to_euclidean(θ, φ)[3] for θ in lat, φ in lon]
mesh_pts = [spherical_to_euclidean(θ, φ) for θ in lat, φ in lon]
data = [F(spherical_to_euclidean(θ, φ)) for θ in lat, φ in lon]
# Init contour plot
scene = Scene()
mesh!(
    scene,
    Makie.Sphere(Point3f0(0), 0.9975f0),
    color = data,
    colormap = ColorSchemes.viridis,
    levels = 50,
    interpolate = false,
    show_axis = false,
)
#
# Run PSO
# (a) equiangular x0
N = 50
x0 = [random_point(M) for i in 1:200]
xPlot = deepcopy(x0)
v0 = [zero_tangent_vector(M, y) for y in x0]
o = particle_swarm(
    M,
    F;
    x0 = x0,
    velocity = v0,
    inertia = 0.05,
    social_weight = 0.3,
    cognitive_weight = 0.2,
    stopping_criterion = StopAfterIteration(800),
    debug = [:Iteration, "\n", 100],
    record = [:x, :p, :velocity],
    return_options = true,
);
path = [x[1] for x in get_record(o, :x)] # x0
velocities = [v[1] for v in get_record(o, :velocity)] # x0

# Animate / record
t = Node(xPlot) # create a life signal
b = Node(xPlot[argmin([F(y) for y in xPlot])])

scatter!(# plot population
    scene,
    lift(t -> [x[1] for x in t], t),
    lift(t -> [x[2] for x in t], t),
    lift(t -> [x[3] for x in t], t),
    markersize = 0.025,
)
# plot best
scatter!(#plot best
    scene,
    lift(b -> [b[1]], b),
    lift(b -> [b[2]], b),
    lift(b -> [b[3]], b),
    markersize = 0.033,
    color = :white,
)
N = length(path)
record(scene, string(@__DIR__, "/record_video.mp4"), 1:N, framerate = 30) do i
    t[] = path[i]
    return b[] = path[i][argmin([F(y) for y in path[i]])]
end
best = get_solver_result(o)
