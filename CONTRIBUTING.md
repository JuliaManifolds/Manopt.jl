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
    - [Concerning the documentation](#Concerning-the-documentation)
    - [Spell checking](#Spell-checking)
## I just have a question

The developer can most easily be reached in the Julia Slack channel [#manifolds](https://julialang.slack.com/archives/CP4QF0K5Z).
You can apply for the Julia Slack workspace [here](https://julialang.org/slack/) if you haven't joined yet.
You can also ask your question on [discourse.julialang.org](https://discourse.julialang.org).

## How can I file an issue?

If you found a bug or want to propose a feature, please open an issue in within the [GitHub repository](https://github.com/JuliaManifolds/Manopt.jl/issues).

## How can I contribute?

### Add a missing method

There is still a lot of methods that can be contributed within the optimization framework of [`Manopt.jl`](https://juliamanifolds.github.io/Manopt.jl/),
may it be functions, gradients, differentials, proximal maps, step size rules or stopping criteria.
If you notice a method you could contribute or improve an implementation, please do so,
and the maintainers try help with the necessary details.
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

Try to follow the [documentation guidelines](https://docs.julialang.org/en/v1/manual/documentation/) from the Julia documentation as well as oriented on [Blue Style](https://github.com/invenia/BlueStyle).
[`Manopt.jl`](https://juliamanifolds.github.io/Manopt.jl/) uses [Runic.jl](https://github.com/fredrikekre/Runic.jl) for code formatting.

Please follow a few internal conventions:

- It is preferred that any subtype of a `AbstractManoptProblem`'s struct contains information about the general structure of the problem.
- Any implemented function should be accompanied by its mathematical formulae if a closed form exists.
- `AbstractManoptProblem` and helping functions are stored within the `plan/` folder and sorted by properties of the problem and/or solver at hand.
- the solver state is usually stored with the solver itself
- Within the source code of one algorithm, following the state, the high level interface should be next, then the initialization, then the step.
- Otherwise an alphabetical order of functions is preferable.
- The preceding implies that the mutating variant of a function follows the non-mutating variant.
- Always add a newline between things of different types (struct/method/const).
- Always add a newline between methods for different functions (including mutating/nonmutating variants).
- Prefer to have no newline between methods for the same function; when reasonable, merge the documentation strings.
  You can also define a string for the common documentation and interpolate it in the docstrings of the methods.
- All `import`/`using`/`include` should be in the main module file.

### Concerning the documentation

- if possible provide both mathematical formulae and literature references using [DocumenterCitations.jl](https://juliadocs.org/DocumenterCitations.jl/stable/) and BibTeX where possible
- Always document all input variables, positional arguments with their defaults, and keyword arguments also with their types and default values.
- if applicable, use [DocumenterInterlinks.jl](https://juliadocs.org/DocumenterInterLinks.jl/stable/) when mentioning functions from other packages in the documentation. If you add a reference to a function from a new package, maybe consider adapting the CSS as well to prefix the links with the package logo.
- Write a short entry in the [Changelog.md](https://manoptjl.org/stable/changelog/) to document your changes.

If you implement a new feature, a tutorial how to use it would be appreciated as well. Tutorials are written as [Quarto](https://quarto.org/) documents and stored in the `tutorials/` folder. This is rendered automatically into the documentation page, you just have to add a menu entry within the tutorial sub menu.

If you implement an algorithm with a certain numerical example in mind, it would be great, if this could be added to the [ManoptExamples.jl](https://github.com/JuliaManifolds/ManoptExamples.jl) package as well.

### Spell checking

We use [crate-ci/typos](https://github.com/crate-ci/typos) for spell checking, which is run automatically on GitHub Actions, but you can also run it locally using their command line tool.

### On the use of AI

Following the [Julia Discourse Guidelines – Keep it tidy](https://discourse.julialang.org/faq#keep-tidy),
please do not open PRs or issues that are pure AI generated. `Manopt.jl` is in its aspects,
especially the code, the documentation, as well as the tests, carefully curated to be concise,
well-documented, comprehensive, but in tests also in a (hopefully) good balance between
ensuring functionality and “over testing”.

Of course it is ok to get help from an AI, e.g. when refactoring parts of the code,
but please always carefully reflect on the results proposed and do not “[vibe code](https://en.wikipedia.org/wiki/Vibe_coding)”. That usually does not work well nor fit the exact mathematical definitions,
reliability and stability as well as and abstractions of the provided algorithms `Manopt.jl` aims to provide.