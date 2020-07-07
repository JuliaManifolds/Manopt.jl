using Manopt, Manifolds, Makie, Colors, LinearAlgebra, ColorSchemes

#
# Helpers and Function
rot_x(φ) = [1 0 0; 0 cos(φ) -sin(φ); 0 sin(φ) cos(φ)]
rot_y(φ) = [cos(φ) 0 -sin(φ); 0 1 0; sin(φ)  0 cos(φ)]
rot_z(φ) = [cos(φ) -sin(φ) 0;sin(φ) cos(φ) 0; 0 0 1]
A = Diagonal([2.0, 1.0, 0.5])
B = rot_z(pi/4)*Diagonal([4.0, 1.0, 2.0])rot_z(-π/4)
C = rot_x(π/8)*rot_y(-π/4)*Diagonal([0.5, 2.0, 3.0])*rot_y(π/4)*rot_x(-π/8)
F(x) = x'*(A+B-C)*x
M = Manifolds.Sphere(2)

#
# Plot Settings
n_lon = 401
n_lat = 201
lat = range(0, π; length=n_lat)
lon = range(-π, π; length=n_lon)
sphere_color = colorant"rgb(223 186 105)"
plane_color = colorant"rgb(255 254 223)"
manopt_pre_scheme = ColorScheme([sphere_color,plane_color], "Manopt", "The Sphere color Scheme for the Manopt Logo")
manopt_scheme = ColorScheme([get(manopt_pre_scheme, i) for i in 0.0:0.01:1.0], "Manopt", "The Sphere color Scheme for the Manopt Logo")
spherical_to_euclidean(θ,φ,r=1) = [r*sin(θ)*cos(φ), r*sin(θ)*sin(φ), r*cos(θ)]
mesh_x = [spherical_to_euclidean(θ,φ)[1] for θ ∈ lat, φ ∈ lon]
mesh_y = [spherical_to_euclidean(θ,φ)[2] for θ ∈ lat, φ ∈ lon]
mesh_z = [spherical_to_euclidean(θ,φ)[3] for θ ∈ lat, φ ∈ lon]
mesh_pts = [spherical_to_euclidean(θ,φ) for θ ∈ lat, φ ∈ lon]
data = [F(spherical_to_euclidean(θ,φ)) for θ ∈ lat, φ ∈ lon]
# Init contour plot
scene = Scene()
mesh!(scene, Makie.Sphere(Point3f0(0), 0.99f0), color=data, colormap = ColorSchemes.viridis, levels=50, interpolate=false, show_axis = false)

#
# Run PSO
x0 = [random_point(M) for i ∈ 1:50]
# o = particle_swarm(M,F,x0; record = [:x])
# path = get_record(o)
path = [ [rot_y(i*π/360)*rot_z(1*i*π/180)*x for x in x0] for i=1:360]

# Animate / record
t = Node(path[1]) # create a life signal

scatter!(# plot population
    scene,
    lift(t->[x[1] for x in t],t),
    lift(t -> [x[2] for x in t],t),
    lift(t->[x[3] for x in t],t),
    markersize = 0.05
)
# plot personal best per point
# plot global best
N = length(path)
record(scene, string(@__DIR__, "/record_video.mp4"), 1:N) do i
    t[] = path[i]
end
