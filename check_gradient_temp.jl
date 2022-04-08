using Manifolds, Manopt, Plots

M = Sphere(10)
q = zeros(11);
q[1] = 1.0;
p = zeros(11);
p[1:4] .= 1 / sqrt(4)

r = log(M, p, q)

F(M, p) = 1 / 2 * distance(M, p, q)^2
gradF(M, p) = -log(M, p, q)

check_gradient(M, F, gradF, p, r; plot=true, io=stdout)
