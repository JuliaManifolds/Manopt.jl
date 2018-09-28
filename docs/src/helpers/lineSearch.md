```@meta
CurrentModule = Manopt
```
# [Line Search Functions](@id LineSearchSection)
In several solvers one requires to do a line search. THe algorithm is implemented
as
```@docs
ArmijoLineSearch
getStepsize
```
using the following `<:Options`.
```@docs
LineSearchOptions
SimpleLineSearchOptions
ArmijoLineSearchOptions
```
