### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 3e19b659-8855-4020-9173-fe4a1ffbbc08
using Pkg; Pkg.activate();

# ╔═╡ eb35384f-0822-4953-8d34-4ed5a0df71c5
begin
	using LinearAlgebra
	using RandomMatrices
	using Manopt
	using ManoptExamples
	using Manifolds
	using Random
	using WGLMakie, Makie, GeometryTypes, Colors
end;

# ╔═╡ 587cf35f-882f-4e5a-8527-a53ba7c5644f
begin
    #Random.seed!(42)
    N = 10

    # Eigenwerte vorgeben und orthogonal transformieren:

    #Diagonalmatrix mit Eigenwerten:
    D = zeros(N,N)
    for i in 1:N
        D[i, i] = i
    end

    # Zufällige orthogonale Matrix:
    O = rand(Haar(1),N)

    # Transformation von D:
    rayleigh_A = O' * D * O
	#A = D
    #for i in 1:n
		#for j in 1:n
		#	if i != j
		#		A[i,j] = 0.5
		#	end
		#end
	#end
	println(eigvals(rayleigh_A))
end;

# ╔═╡ 1ce0e60b-ac33-482d-ba5f-fd7b53b79fe8
function f_prime(p)
    return (2.0 * rayleigh_A * p)'
end

# ╔═╡ cd8d3a84-c7bd-448b-9427-1140fbb8b7f1
function f_second_derivative(p)
    return 2.0 * rayleigh_A
end

# ╔═╡ e16a578c-3d64-11f0-057c-c1f978ba732a
begin
struct NewtonEquation{F, T, NM, Nrhs}
	f_prime::F
	f_second_prime::T
	A::NM
	b::Nrhs
end

function NewtonEquation(M, f_pr, f_sp)
	n = manifold_dimension(M)
	A = zeros(N+1,N+1)
	b = zeros(N+1)
	return NewtonEquation{typeof(f_pr), typeof(f_sp), typeof(A), typeof(b)}(f_pr, f_sp, A, b)
end
	
function (ne::NewtonEquation)(M, VB, p)
    ne.A .= hcat(vcat(ne.f_second_prime(p) - ne.f_prime(p)*p*Matrix{Float64}(I, N, N), p'), vcat(p, 0))
    ne.b .= vcat(ne.f_prime(p)', 0)
	return
end
	
function (ne::NewtonEquation)(M, VB, p, p_trial)
    return vcat(vector_transport_to(M, p, ne.f_prime(p_trial)', p_trial,  ProjectionTransport()), 0)
end
end;

# ╔═╡ 7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
function solve_augmented_system(problem, newtonstate) 
	res = (problem.newton_equation.A) \ (-problem.newton_equation.b)
	return res[1:N]
end;

# ╔═╡ 427233d0-567d-4cac-930d-7c80178f2d09
begin
	M = Manifolds.Sphere(N-1)
	y0 = copy(M, 1/sqrt(N)*ones(N))
	
	NE = NewtonEquation(M, f_prime, f_second_derivative)
		
	st_res = vectorbundle_newton(M, TangentBundle(M), NE, y0; sub_problem=solve_augmented_system, sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(150)|StopWhenChangeLess(M,1e-11; outer_norm=Inf)),
	retraction_method=ProjectionRetraction(),
	#stepsize=Manopt.AffineCovariantStepsize(M, theta_des=0.1),
	#stepsize=ConstantLength(power, 1.0),
	debug=[:Iteration, (:Change, "Change: %1.8e"), "\n", :Stop, (:Stepsize, "Stepsize: %1.8e"), "\n",],
	record=[:Iterate, :Change],
	return_state=true
)
end

# ╔═╡ 66bb735a-a41d-4742-8581-fe48907b7489
change = get_record(st_res, :Iteration, :Change)[2:end];

# ╔═╡ 285c7cdc-fdc7-43e8-99fa-17fd708d4328
begin
	f = Figure(;)
	
    row, col = fldmod1(1, 2)
	
	Axis(f[row, col], yscale = log10, title = string("Semilogarithmic Plot of the norms of the Newton direction"), xminorgridvisible = true, xticks = (1:length(change)), xlabel = "Iteration", ylabel = "‖δx‖")
    scatterlines!(change[1:end], color = :blue)
	f
end

# ╔═╡ dd7fea0d-9ada-4bec-91c8-2338969f71e6
res = get_record(st_res, :Iteration, :Iterate)[end]

# ╔═╡ 4c756d92-6e43-4bc8-8f84-35373ba8172b
res'*rayleigh_A*res

# ╔═╡ Cell order:
# ╠═3e19b659-8855-4020-9173-fe4a1ffbbc08
# ╠═eb35384f-0822-4953-8d34-4ed5a0df71c5
# ╠═587cf35f-882f-4e5a-8527-a53ba7c5644f
# ╠═1ce0e60b-ac33-482d-ba5f-fd7b53b79fe8
# ╠═cd8d3a84-c7bd-448b-9427-1140fbb8b7f1
# ╠═e16a578c-3d64-11f0-057c-c1f978ba732a
# ╠═7bf95fe2-20cb-4824-9d5d-b520b4e6e1ad
# ╠═427233d0-567d-4cac-930d-7c80178f2d09
# ╠═66bb735a-a41d-4742-8581-fe48907b7489
# ╠═285c7cdc-fdc7-43e8-99fa-17fd708d4328
# ╠═dd7fea0d-9ada-4bec-91c8-2338969f71e6
# ╠═4c756d92-6e43-4bc8-8f84-35373ba8172b
