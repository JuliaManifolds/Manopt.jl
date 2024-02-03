# Contributing to `Manopt.jl`

First, thanks for taking the time to contribute.
Any contribution is appreciated and welcome.

The following is a set of guidelines to [`Manopt.jl`](https://juliamanifolds.github.io/Manopt.jl/).

#### Table of contents

- [Contributing to `Manopt.jl`](#Contributing-to-manoptjl)
      - [Table of Contents](#Table-of-Contents)
  - [I just have a question](#I-just-have-a-question)
  - [How can I file an issue?](#How-can-I-file-an-issue)
  - [How can I contribute?](#How-can-I-contribute)
    - [Add a missing method](#Add-a-missing-method)
    - [Provide a new algorithm](#Provide-a-new-algorithm)
    - [Provide a new example](#Provide-a-new-example)
    - [Code style](#Code-style)

## I just have a question

The developer can most easily be reached in the Julia Slack channel [#manifolds](https://julialang.slack.com/archives/CP4QF0K5Z).
You can apply for the Julia Slack workspace [here](https://julialang.org/slack/) if you haven't joined yet.
You can also ask your question on [discourse.julialang.org](https://discourse.julialang.org).

## How can I file an issue?

If you found a bug or want to propose a feature, please open an issue in within the [GitHub repository](https://github.com/JuliaManifolds/Manopt.jl/issues).

## How can I contribute?

### Add a missing method

There is still a lot of methods for within the optimization framework of  `Manopt.jl`, may it be functions, gradients, differentials, proximal maps, step size rules or stopping criteria.
If you notice a method missing and can contribute an implementation, please do so, and the maintainers will try help with the necessary details.
Even providing a single new method is a good contribution.

### Provide a new algorithm

A main contribution you can provide is another algorithm that is not yet included in the
package.
An algorithm is always based on a concrete type of a [`AbstractManoptProblem`](https://manoptjl.org/stable/plans/index.html#AbstractManoptProblems-1) storing the main information of the task and a concrete type of an [`AbstractManoptSolverState`](https://manoptjl.org/stable/plans/index.html#AbstractManoptSolverState-1from) storing all information that needs to be known to the solver in general. The actual algorithm is split into an initialization phase, see [`initialize_solver!`](https://manoptjl.org/stable/solvers/index.html#Manopt.initialize_solver!), and the implementation of the `i`th step of the solver itself, see  before the iterative procedure, see [`step_solver!`](https://manoptjl.org/stable/solvers/index.html#Manopt.step_solver!).
For these two functions, it would be great if a new algorithm uses functions from the [`ManifoldsBase.jl`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html) interface as generically as possible. For example, if possible use [`retract!(M,q,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.retract!-Tuple{AbstractManifold,Any,Any,Any}) in favor of [`exp!(M,q,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.exp!-Tuple{AbstractManifold,Any,Any,Any}) to perform a step starting in `p` in direction `X` (in place of `q`), since the exponential map might be too expensive to evaluate or might not be available on a certain manifold. See [Retractions and inverse retractions](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#Retractions-and-inverse-Retractions) for more details.
Further, if possible, prefer [`retract!(M,q,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.retract!-Tuple{AbstractManifold,Any,Any,Any}) in favor of [`retract(M,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.retract-Tuple{AbstractManifold,Any,Any}), since a computation in place of a suitable variable `q` reduces memory allocations.

Usually, the methods implemented in `Manopt.jl` also have a high-level interface, that is easier to call, creates the necessary problem and options structure and calls the solver.

The two technical functions `initialize_solver!` and `step_solver!` should be documented with technical details, while the high level interface should usually provide a general description and some literature references to the algorithm at hand.

### Provide a new example

Example problems are available at [`ManoptExamples.jl`](https://github.com/JuliaManifolds/ManoptExamples.jl),
where also their reproducible Quarto-Markdown files are stored.

### Code style

We try to follow the [documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/) from the Julia documentation as well as [Blue Style](https://github.com/invenia/BlueStyle).
We run [`JuliaFormatter.jl`](https://github.com/domluna/JuliaFormatter.jl) on the repository in the way set in the `.JuliaFormatter.toml` file, which enforces a number of conventions consistent with the Blue Style. We further run [vale](https://vale.sh) on both Markdown files and code files, affecting documentation and source code comments

We also follow a few internal conventions:

- It is preferred that the `AbstractManoptProblem`'s struct contains information about the general structure of the problem.
- Any implemented function should be accompanied by its mathematical formulae if a closed form exists.
- `AbstractManoptProblem` and helping functions are stored within the `plan/` folder and sorted by properties of the problem and/or solver at hand.
- the solver state is usually stored with the solver itself
- Within the source code of one algorithm, following the state, the high level interface should be next, then the initialization, then the step.
- Otherwise an alphabetical order of functions is preferable.
- The preceding implies that the mutating variant of a function follows the non-mutating variant.
- There should be no dangling `=` signs.
- Always add a newline between things of different types (struct/method/const).
- Always add a newline between methods for different functions (including mutating/nonmutating variants).
- Prefer to have no newline between methods for the same function; when reasonable, merge the documentation strings.
- All `import`/`using`/`include` should be in the main module file.

Concerning documentation

- if possible provide both mathematical formulae and literature references using [DocumenterCitations.jl](https://juliadocs.org/DocumenterCitations.jl/stable/) and BibTeX where possible
- Always document all input variables and keyword arguments

If you implement an algorithm with a certain application in mind, it would be great, if this could be added to the [ManoptExamples.jl](https://github.com/JuliaManifolds/ManoptExamples.jl) package as well.