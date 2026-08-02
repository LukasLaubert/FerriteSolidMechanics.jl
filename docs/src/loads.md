# External loads

`LoadHandler` assembles the external force vector of a simulation.
The code lives in `src/Loads.jl`.

This page understands loads as Neumann boundary conditions, i.e., prescribed forces.
Prescribed displacements are Dirichlet boundary conditions and are handled by Ferrite's `Dirichlet` and `ConstraintHandler`.

## Load types

Each load type is named after the units of the value its function returns.

| Type | Units | Sets | Applies |
|---|---|---|---|
| [`BodyForce`](@ref) | force/volume in 3D, force/area in 2D | cell sets, or all cells | the returned vector |
| [`Traction`](@ref) | force/area in 3D, force/length in 2D | facet sets | the returned vector |
| [`Pressure`](@ref) | same as `Traction` | facet sets | `-f * n` – positive acts inward along normal `n` |
| [`NodalForce`](@ref) | force | node sets | the returned vector at each node |

`Traction` covers area loads in 3D and edge loads in 2D without requiring a separate type: a facet is a face in 3D and an edge in 2D.
Line loads along an edge of a 3D cell are not currently available.

## Building a handler

`LoadHandler` is built and used the way `ConstraintHandler` is:

```julia
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0], [1, 2]))
close!(ch)

lh = LoadHandler(assembler)
add!(lh, BodyForce(x -> Vec(0.0, -rho * g)))
add!(lh, Traction("top", (x, t) -> Vec(0.0, -p * t)))
add!(lh, Pressure("hole", (x, t) -> p * t))
add!(lh, NodalForce("tip", x -> Vec(0.0, -F); distribute=true))
close!(lh)
```

The strings `"top"`, `"hole"` and `"tip"` are Ferrite grid set names.
They are checked against the grid once the handler is closed using `close!`.
Alternatively, a set object is accepted as well, so `Traction("top", f)` and `Traction(getfacetset(grid, "top"), f)` are equivalent.

`LoadHandler(assembler)` reads the `DofHandler`, the `thickness` and the quadrature orders from the assembler, so external and internal forces are integrated consistently.
This matters in 2D, where facet and body integrals scale with `thickness` exactly as `ke` and `re` do.
The bare `LoadHandler(dh)` form is for drivers that assemble no internal forces and defaults to `thickness = 1.0`.

`close!` throws an `ArgumentError`, naming the entry at fault, for an unknown or empty set, a facet or node carrying no degrees of freedom, a load function whose call signature matches none of the accepted ones, or a function returning the wrong shape.

## Load functions

A load function may be defined as `f(x)`, `f(x, t)` or `f(x, t, n)`, and `close!` selects the signature the function actually defines.
`x` is the spatial coordinate and `n` is the outward facet normal, available on `Traction` and `Pressure` only.
`t` is the value handed to [`external_forces!`](@ref) in the time loop, as shown in [Placement in the Newton loop](#Placement-in-the-Newton-loop) below.

Since `f` receives `x` and `t`, spatial and temporal profiles are written directly in the load function:

```julia
Traction("top", x -> Vec(0.0, -p))                                 # uniform, constant in t
Traction("top", (x, t) -> Vec(0.0, -p * t))                        # uniform, ramped in t
Traction("top", x -> Vec(0.0, -p * x[1] / L))                      # triangular along x, constant in t
Traction("top", (x, t) -> Vec(0.0, -p * (x[1] / L) * t))           # triangular along x, ramped in t
Traction("top", (x, t) -> Vec(0.0, -p * (x[1] / L)^2 * sin(w*t)))  # quadratic in x, sinusoidal in t
Pressure("dam", x -> rho * g * max(H - x[2], 0.0))                 # hydrostatic, zero above the free surface at y = H
BodyForce(x -> Vec(0.0, -rho * g))                                 # self-weight
BodyForce("rotor", x -> rho * omega^2 * x)                         # centrifugal about the origin
```

`t` carries the meaning the calling code gives it: physical time for rate-dependent materials, or a load factor for quasi-static ramping.
Give the same value to Ferrite's `update!(ch, t)` so that prescribed displacements and loads refer to the same instant.

`n` is evaluated on the undeformed geometry.
A normal-aligned traction is therefore a dead load, i.e., fixed in direction at the start and not rotating as the surface deforms.
Follower loads, whose direction tracks the deforming surface, are not currently available.

## Placement in the Newton loop

`stiffness_matrix` returns the **internal** force vector, so the residual is formed in your Newton loop:

```julia
f_ext = external_forces!(lh, t)
for iter in 1:max_newton
    K, f_int = stiffness_matrix(assembler, u; dt=dt)
    residual .= f_int .- f_ext
    apply_zero!(K, residual, ch)
    norm(residual) <= tol * max(norm(f_ext), 1.0) && break
    u .-= K \ residual
end
```

Three points follow from this:

- Dead loads do not depend on the displacement, so `external_forces!` is called **once per step, outside the Newton loop**.
- External loads contribute **nothing to `K`**.
- Scaling the convergence threshold with `norm(f_ext)` makes it relative to the applied load.
  The `max(..., 1.0)` covers pure displacement control for cases where `norm(f_ext)` is zero.

Under adaptive time stepping, a rejected step changes the trial time, so `external_forces!` is called again with the new value before the step is retried.

`external_forces!` writes into a vector belonging to the `LoadHandler` and returns it.
Calling `external_forces!` again overwrites that vector, so copy the result when a step's values are needed afterwards.

## Changing the loading during a simulation

Changing the magnitude, direction or profile of a load needs no further call to `close!`, because the load function is evaluated on every `external_forces!` call.

Any profile that can be written in terms of `t` works directly, including switching a load off:

```julia
# zero(Vec{2}) is the null traction of a 2D problem; use Vec{3} in 3D
Traction("top", (x, t) -> t < t_hold ? Vec(0.0, -p * t) : zero(Vec{2}))  # ramp, then release
Traction("top", (x, t) -> Vec(0.0, -p * sin(w * t)))                     # cyclic, sign reversal included
```

Loading that depends on future conditions rather than on `t` alone is written by closing over a `Ref`:

```julia
stage = Ref(1)
add!(lh, Traction("top", (x, t) -> stage[] == 1 ? Vec(0.0, -p) : zero(Vec{2})))
# inside the time loop; the next external_forces! call picks up the new value
peak_stress > sigma_crit && (stage[] = 2)
```

The same pattern removes a load, since a function returning zero contributes nothing to the external force vector.

A load on a set that was never declared is the only case requiring another `close!(lh)` call.
This prepares the entries added since the last call and leaves earlier ones untouched:

```julia
add!(lh, Traction("second_stage_face", (x, t) -> Vec(0.0, -q * (t - t_switch))))
close!(lh)
```

## Replication under MPI

`LoadHandler` assembles serially and, under MPI, redundantly: every rank walks every node, facet, and cell, and builds the complete external force vector.
No reduction is performed, and every rank must add the same loads in the same order so that the per-rank copies agree.

This matches the replicated-data design described on the [Concepts](concepts.md) page, where `stiffness_matrix` has already reduced the internal force so that it is complete on every rank as well.
Adding an `MPI.Allreduce` over `f_ext` would sum those identical copies and apply the load `nranks` times.

## Distributing a resultant

`NodalForce(set, f)` applies the force returned by `f` at every node of the set, which is the default `distribute=false`.
With `distribute=true` that force is instead read as the resultant of the whole set and split equally over its nodes, so the sum over the set is the force itself.

Equal splitting matches the finite element load distribution only for first-order shape functions.
Above first order the shape function integration no longer splits the load equally, so a uniform `Traction` on a quadratic mesh leads to different shares at corner and midside nodes.
To spread a resultant over a surface on such a mesh, use `Traction` with magnitude `F / A`.
