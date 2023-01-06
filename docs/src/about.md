# About

Manopt.jl inherited its name from [Manopt](https://manopt.org), a Matlab toolbox for optimization on manifolds.
This Julia package was started and is currently maintained by [Ronny Bergmann](https://ronnybergmann.net/about.html).

The following people contributed
* [Constantin Ahlmann-Eltze](https://const-ae.name) implemented the [gradient and differential check functions](helpers/checks.md)
* [Renée Dornig](https://github.com/r-dornig) implemented the [particle swarm](@ref ParticleSwarmSolver), the [Riemannian Augmented Lagrangian Method](@ref AugmentedLagrangianSolver), the [Exact Penalty Method](@ref ExactPenaltySolver), as well as the [`NonmonotoneLinesearch`](@ref)
* [Willem Diepeveen](https://www.maths.cam.ac.uk/person/wd292) implemented the [primal-dual Riemannian semismooth Newton](@ref PDRSSNSolver) solver.
* Even Stephansen Kjemsås contributed to the implementation of the [Frank Wolfe Method](@ref FrankWolfe)
* [Tom-Christian Riemer](https://www.tu-chemnitz.de/mathematik/wire/mitarbeiter.php) Riemer implemented the [trust regions](@ref trust_regions) and [quasi Newton](solvers/quasi_Newton.md) solvers.
* [Manuel Weiss](https://scoop.iwr.uni-heidelberg.de/author/manuel-weiß/) implemented most of the [conjugate gradient update rules](@ref cg-coeffs)

...as well as various [contributors](https://github.com/JuliaManifolds/Manopt.jl/graphs/contributors) providing small extensions, finding small bugs and mistakes and fixing them by opening [PR](https://github.com/JuliaManifolds/Manopt.jl/pulls)s.

If you want to contribute a manifold or algorithm or have any questions, visit
the [GitHub repository](https://github.com/JuliaManifolds/Manopt.jl/)
to clone/fork the repository or open an issue.
