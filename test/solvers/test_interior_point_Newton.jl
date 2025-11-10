using Manifolds, Manopt, LinearAlgebra, Random, Test, RecursiveArrayTools

@testset "Interior Point Newton Solver" begin
    @testset "StepsizeState" begin
        M = Manifolds.Sphere(2)
        a = StepsizeState(M)
        b = StepsizeState(a.p, a.X)
        @test a.p === b.p
        @test a.X === b.X
    end
    @testset "A solver run on the Sphere" begin
        # We can take a look at debug prints of one run and plot the result
        # on CI and when running with ] test Manopt, both have to be set to false.
        _debug_iterates_plot = false
        _debug = false

        A = -[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 2.0]
        f(M, p) = 0.5 * p' * A * p
        grad_f(M, p) = (I - p * p') * A * p
        Hess_f(M, p, X) = A * X - (p' * A * X) * p - (p' * A * p) * X

        g(M, p) = -p
        grad_g(M, p) = [(p * p' - I)[:, i] for i in 1:3]
        Hess_g(M, p, X) = [(X * p')[:, i] for i in 1:3]
        M = Manifolds.Sphere(2)

        # With dummy closed form solution
        ipnsc = InteriorPointNewtonState(
            M, ConstrainedManifoldObjective(f, grad_f; g = g, grad_g = grad_g, M = M), f
        )
        @test ipnsc.sub_state isa Manopt.ClosedFormSubSolverState

        p_0 = (1.0 / (sqrt(3.0))) .* [1.0, 1.0, 1.0]
        # p_0 = 1.0 / sqrt(2) .* [0.0, 1.0, 1.0]
        p_opt = [0.0, 0.0, 1.0]
        record = [:Iterate]
        dbg = [
            :Iteration,
            " ",
            :Cost,
            " ",
            :Stepsize,
            " ",
            :Change,
            " ",
            :Feasibility,
            "\n",
            :Stop,
            10,
        ]

        sc = StopAfterIteration(800) | StopWhenKKTResidualLess(1.0e-2)
        # (a) classical call w/ recording
        res = interior_point_Newton(
            M,
            f,
            grad_f,
            Hess_f,
            p_0;
            g = g,
            grad_g = grad_g,
            Hess_g = Hess_g,
            stopping_criterion = sc,
            debug = _debug ? dbg : [],
            record = _debug_iterates_plot ? record : [],
            return_state = true,
            return_objective = true,
        )

        q = get_solver_result(res)
        @test distance(M, q, [0.0, 0.0, 1.0]) < 2.0e-4

        # (b) inplace call
        q2 = copy(M, p_0)
        interior_point_Newton!(
            M,
            f,
            grad_f,
            Hess_f,
            q2;
            g = g,
            grad_g = grad_g,
            Hess_g = Hess_g,
            stopping_criterion = sc,
        )
        @test q == q2

        # (c) call with objective - but then we also test the Centrality cond
        coh = ConstrainedManifoldObjective(
            f, grad_f, g, grad_g, nothing, nothing; hess_f = Hess_f, hess_g = Hess_g, M = M, p = p_0
        )
        ipcc = InteriorPointCentralityCondition(coh, 0.9)
        q3 = interior_point_Newton(
            M, coh, p_0; stopping_criterion = sc, centrality_condition = ipcc
        )
        @test distance(M, q3, [0.0, 0.0, 1.0]) < 2.0e-4
        if _debug_iterates_plot
            using GLMakie, Makie, GeometryTypes
            rec = get_record(res[2])
            prepend!(rec, [p_0])
            add_scale = 0.0075
            rec .+= add_scale * rec # scale slighly to lie on the sphere
            n = 30
            π1(x) = x[1]
            π2(x) = x[2]
            π3(x) = x[3]
            h(x) = [cos(x[1])sin(x[2]), sin(x[1])sin(x[2]), cos(x[2])]
            U = [[θ, ϕ] for θ in LinRange(0, 2π, n), ϕ in LinRange(0, π, n)]
            V = [[θ, ϕ] for θ in LinRange(0, π / 2, n), ϕ in LinRange(0, π / 2, n)]

            pts, pts_ = h.(U), h.(V)
            f_ = p -> f(M, p)
            s = maximum(f_.(pts)) - minimum(f_.(pts))
            s_ = maximum(f_.(pts_)) - minimum(f_.(pts_))
            x1, x2, x3, x1_, x2_ = π1.(pts), π2.(pts), π3.(pts), π1.(pts_), π2.(pts_)
            x3_ = π3.(pts_)
            grads = grad_f.(Ref(M), pts)
            normgrads = grads ./ norm.(grads)
            v1, v2, v3 = π1.(normgrads), π2.(normgrads), π3.(normgrads)

            scene = Scene()
            cam3d!(scene)
            range_f = (minimum(f_.(pts)), maximum(f_.(pts)))

            pa = [:color => f_.(pts), :backlight => 1.0f0, :colorrange => range_f]
            # light colormap on sphere
            surface!(scene, x1, x2, x3; colormap = (:viridis, 0.4), pa...)
            # ful color on feasible set
            surface!(scene, x1_, x2_, x3_; colormap = (:viridis, 1.0), backlight = 1.0f0, pa...)
            scatter!(scene, π1.(rec), π2.(rec), π3.(rec); color = :black, markersize = 8)
            P = [(1 + add_scale) .* p_opt]
            scatter!(scene, π1.(P), π2.(P), π3.(P); color = :white, markersize = 9)
            display(scene)
        end
    end
end
