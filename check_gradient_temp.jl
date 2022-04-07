using Manifolds, Manopt, Plots

M = Sphere(10)
q = zeros(11);
q[1] = 1.0;
p = zeros(11);
p[1:2] .= 1 / sqrt(2)

F(M, p) = 1 / 2 * distance(M, p, q)^2
gradF(M, p) = -log(M, p, q)

check_gradient(M, F, gradF, p, log(M, p, q); plot=true, io=stdout)
