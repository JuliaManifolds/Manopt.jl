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

    Random.seed!(56343658)

    N = 101

    # Eigenwerte vorgeben und orthogonal transformieren:

    #Diagonalmatrix mit Eigenwerten:
    eig_A = Matrix{Float64}(I, N, N)
    for i in 1:N
        eig_A[i, i] = i
    end

    # Zufällige orthogonale Matrix:
    O = rand(Haar(1),N)

    # Transformation von D:
    #A = O' * D * O
	
    for i in 1:N
		for j in 1:N
		if i > j
			eig_A[i,j] = 1.0
			end
			if i < j
			eig_A[i,j] = 1.0
		end
	end
	end
	B = zeros(N,N)
	for i in 1:N
		for j in 1:N
			if i > j
				B[i,j] = -1.0
			end
			if i < j
				B[i,j] = 0.0
			end
			if i == j
				B[i, j] = 0.0
			end
		end
	end
	println(eigvals(eig_A))
	println(eigvals(B))
end;

# ╔═╡ 1ce0e60b-ac33-482d-ba5f-fd7b53b79fe8
function f_prime(p)
    return (eig_A * p)'
end

# ╔═╡ cd8d3a84-c7bd-448b-9427-1140fbb8b7f1
function f_second_derivative(p)
    return eig_A
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
    ne.A .= hcat(vcat(ne.f_second_prime(p) - 1/(norm(B*p)^2) * (ne.f_prime(p)*(B*p)*B), p'), vcat(B*p, 0))
    ne.b .= vcat(ne.f_prime(p)', 0)
	return
end
	
function (ne::NewtonEquation)(M, VB, p, p_trial)
    return vcat((ne.f_prime(p_trial) - 1/norm(B*p)^2*(B*p)'*(ne.f_prime(p_trial)*(B*p)))', 0)
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
	#stepsize=Manopt.AffineCovariantStepsize(M, theta_des=0.5),
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
res'*eig_A*res

# ╔═╡ a7a13b69-dd2c-4f24-989d-dc3356633184
norm(eig_A*res - res'*eig_A*res*B*res)

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
# ╠═a7a13b69-dd2c-4f24-989d-dc3356633184
