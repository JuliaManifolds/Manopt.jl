# About

Manopt.jl inherited its name from [Manopt](https://manopt.org), a Matlab toolbox for optimisation on manifolds.
This Julia package was started and is currently maintained by [Ronny Bergmann](https://ronnybergmann.net/about.html).

The following people contributed
* [Constantin Ahlmann-Eltze](https://const-ae.name) implemented the [gradient and differential check functions](helpers/checks.md)
* Renée Dornig implemented the [particle swarm](@ref ParticleSwarmSolver) and the [`NonmonotoneLinesearch`](@ref)
* [Tom-Christian Riemer](https://www.tu-chemnitz.de/mathematik/wire/mitarbeiter.php) Riemer implemented the [trust regions](@ref trust_regions) and [quasi Newton](solvers/quasi_Newton.md) solvers.
* [Manuel Weiss](https://scoop.iwr.uni-heidelberg.de/author/manuel-weiß/) implemented most of the [conjugate gradient update rules](@ref cg-coeffs)
* Laura Weigl provided the [robust PCA](examples/robustPCA.md) and [smallest Eigenvalue](examples/smallestEigenvalue.md) example notebooks.
* [Willem Diepeveen](https://www.maths.cam.ac.uk/person/wd292) implemented the [primal-dual Riemannian semismooth Newton](@ref PDRSSNSolver) solver.

...as well as various [contributors](https://github.com/JuliaManifolds/Manopt.jl/graphs/contributors) providing small extensions, finding small bugs and mistakes and fixing them by opening [PR](https://github.com/JuliaManifolds/Manopt.jl/pulls)s.

If you want to contribute a manifold or algorithm or have any questions, visit
the [GitHub repository](https://github.com/JuliaManifolds/Manopt.jl/)
to clone/fork the repository or open an issue.
