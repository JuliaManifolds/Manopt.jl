# Contributing to `Manopt.jl`

First, thanks for taking the time to contribute.
Any contribution is appreciated and welcome.

The following is a set of guidelines to [`Manopt.jl`](https://juliamanifolds.github.io/Manopt.jl/).

#### Table of Contents

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

If you found a bug or want to propose a feature, we track our issues within the [GitHub repository](https://github.com/JuliaManifolds/Manopt.jl/issues).

## How can I contribute?

### Add a missing method

There is still a lot of methods for within the optimisation framework of  `Manopt.jl`, may it be functions, gradients, differentials, proximal maps, step size rules or stopping criteria.
If you notice a method missing and can contribute an implementation, please do so!
Even providing a single new method is a good contribution.

### Provide a new algorithm

A main contribution you can provide is another algorithm that is not yet included in the
package.
An alorithm is always based on a is a concrete type of a [`Problem`](https://manoptjl.org/stable/plans/index.html#Problems-1) storing the main information of the task and a concrete type of an [`Option`](https://manoptjl.org/stable/plans/index.html#Options-1from) storing all information that needs to be known to the solver in general. The actual algorithm is split into an initialization phase, see [`initialize_solver!`](https://manoptjl.org/stable/solvers/index.html#Manopt.initialize_solver!), and the implementation of the `i`th step of the sovler itself, see  before the iterative procedure, see [`step_solver!`](https://manoptjl.org/stable/solvers/index.html#Manopt.step_solver!).
For these two functions it would be great if a new algorithm uses functions from the [`ManifoldsBase.jl`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html) interface as generic as possible. For example, if possible use [`retract!(M,q,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.retract!-Tuple{AbstractManifold,Any,Any,Any}) in favour of [`exp!(M,q,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.exp!-Tuple{AbstractManifold,Any,Any,Any}) to perform a step starting in `p` in direction `X` (in place of `q`), since the exponential map might be too expensive to evaluate or might not be available on a certain manifold. See [Retractions and inverse retractions](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#Retractions-and-inverse-Retractions) for more details.
Further, if possible, prefer [`retract!(M,q,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.retract!-Tuple{AbstractManifold,Any,Any,Any}) in favour of [`retract(M,p,X)`](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html#ManifoldsBase.retract-Tuple{AbstractManifold,Any,Any}), since a computation in place of a suitable variable `q` reduces memory allocations.

Usually, the methods implemented in `Manopt.jl` also have a high-level interface, that is easier to call, creates the necessary problem and options structure and calls the solver.

The two technical functions `initialize_solver!` and `step_solver!` should be documented with technical details, while the high level interface should usually provide a general description and some literature references to the algorithm at hand.

### Provide a new example

The `examples/` folder features several examples covering all solvers. Still, if you have a new example that you implemented yourself for fun or for a paper, feel free to add it to the repository as well. Also if you have a [Pluto](https://github.com/fonsp/Pluto.jl) notebook of your example, feel free to contribute that.

### Code style

We try to follow the [documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/) from the Julia documentation as well as [Blue Style](https://github.com/invenia/BlueStyle).
We run [`JuliaFormatter.jl`](https://github.com/domluna/JuliaFormatter.jl) on the repo in the way set in the `.JuliaFormatter.toml` file, which enforces a number of conventions consistent with the Blue Style.

We also follow a few internal conventions:

- It is preferred that the `Problem`'s struct contains information about the general structure of the problem
- Any implemented function should be accompanied by its mathematical formulae if a closed form exists.
- Problem and option structures are stored within the `plan/` folder and sorted by properties of the problem and/or solver at hand
- Within the source code of one algorithm, the high level interface should be first, then the initialisation, then the step.
- Otherwise an alphabetical order is preferrable.
- The above implies that the mutating variant of a function follows the non-mutating variant.
- There should be no dangling `=` signs.
- Always add a newline between things of different types (struct/method/const).
- Always add a newline between methods for different functions (including mutating/nonmutating variants).
- Prefer to have no newline between methods for the same function; when reasonable, merge the docstrings.
- All `import`/`using`/`include` should be in the main module file.
