using Manopt, Manifolds, Makie, Colors, LinearAlgebra, ColorSchemes

#
# Helpers and Function
rot_x(φ) = [1 0 0; 0 cos(φ) -sin(φ); 0 sin(φ) cos(φ)]
rot_y(φ) = [cos(φ) 0 -sin(φ); 0 1 0; sin(φ)  0 cos(φ)]
rot_z(φ) = [cos(φ) -sin(φ) 0;sin(φ) cos(φ) 0; 0 0 1]
A = Diagonal([2.0, 1.0, 0.5])
B = rot_z(pi/4)*Diagonal([4.0, 1.0, 2.0])rot_z(-π/4)
C = rot_x(π/8)*rot_y(-π/4)*Diagonal([0.5, 2.0, 3.0])*rot_y(π/4)*rot_x(-π/8)
F(x) = x'*(A+B+C)*x
M = Manifolds.Sphere(2)

#
# Plot Settings
n_lon = 201
n_lat = 101
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
mesh!(scene, Makie.Sphere(Point3f0(0), 0.975f0), color=data, colormap = ColorSchemes.viridis, levels=50, interpolate=false)

#
# Run PSO
s1 = 7
s2 = 20
x0 = Array{Array{Float64,1},1}()
for θ ∈ range(0,π,length=s1)
    n = Integer(floor(s2*(1-abs(cos(θ)))))
    if n==0
        push!(x0,spherical_to_euclidean(θ,0.0))
    else
        for φ ∈ range(0, 2π; length = n+1)
            push!(x0,spherical_to_euclidean(θ,φ))
        end
    end
end
# o = particle_swarm(M,F,x0; record = [:x])
# path = get_record(o)
path = [ [rot_y(i*π/360)*rot_z(1*i*π/180)*x for x in x0] for i=1:360]

# Animate / record
t = Node(path[1]) # create a life signal

scatter!(scene, lift(t->[x[1] for x in t],t), lift(t -> [x[2] for x in t],t), lift(t->[x[3] for x in t],t))
N = length(path)
record(scene, "record_video.mp4", 1:N) do i
    t[] = path[i]
end