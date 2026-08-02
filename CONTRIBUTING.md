# Contributing

FerriteSolidMechanics.jl is a solid mechanics assembly framework and material model library built around Ferrite.jl.
Contributions are very welcome, while changes should keep the material interface stable and preserve the trial/commit state semantics documented in `docs/src/developer_guide.md`.

## Licensing and contribution rights

FerriteSolidMechanics.jl is distributed under the MIT license.
By submitting a contribution, you agree that your contribution is licensed under the same MIT license and certify that you have the right to submit it.

For new material models, include enough provenance for future users and maintainers to understand where the implementation came from:

- scientific references for the model,
- whether the implementation is original, translated, or adapted,
- upstream code URL and license, if code-level references were used,
- confirmation that translated or adapted code may be distributed under this repository's MIT license,
- acknowledgements for authors who helped with formulation, parameters, validation, or reference code,
- any upstream copyright, license, or notice text required by the upstream license.

Do not copy code from sources whose license is incompatible with MIT redistribution unless the maintainers explicitly decide to change the licensing model.

## Local checks

Run the test suite before opening a pull request:

```julia
pkg> test
```

or from the repository root:

```sh
julia --project=. test/runtests.jl
```

For new material models, it is recommended to add focused tests that exercise the new model explicitly; see [Material models](#material-models).
The existing test suite does not automatically cover newly added models beyond package loading.
If you are unsure how to add a test, open the pull request anyway and mention this; maintainers can help to add appropriate tests.

For documentation changes, also run:

```sh
julia --project=docs docs/make.jl
```

## Material models

At a glance, a new model implements the quadrature point constitutive interface (`material_response` & `kinematics`), plus the state hooks (`create_state`, `update_state!`, `revert_state!`) for history-dependent models.
Stateless hyperelastic models can instead subtype `AbstractHyperelastic` and implement only the energy `Ψ(C, material)`.
Models that should also run in 2D further implement `compute_PK1_3D` and `update_state_from_3D!` for the `PlaneStrain` / `PlaneStress` wrappers (automatic for `AbstractHyperelastic` subtypes).

The full interface is covered in the Developer Guide, including optional hooks (`set_trial!`, `copy_state!`, tangent-free `_compute_stress_qp`), when to write a custom `_assemble_element!`, and the testing and documentation requirements:

- [The required interface](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/developer_guide/#The-required-interface)
- [Custom element assembly](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/developer_guide/#Custom-element-assembly)
- [Testing material model implementations](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/developer_guide/#Testing-material-model-implementations)

## Naming convention

Please adhere strictly to the **[Naming conventions](https://lukaslaubert.github.io/FerriteSolidMechanics.jl/dev/developer_guide/#Naming-conventions)** detailed in the Developer Guide. This ensures consistent naming across material constructors, state structures, source files, and documentation pages for both canonical and specialized models.

## Style

Whenever possible, prefer the existing code patterns over new abstractions.
Keep changes scoped and use LF line endings for source and docs as configured in `.gitattributes`.