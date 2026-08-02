using Ferrite
using FerriteSolidMechanics
using LinearAlgebra

"""
    quarter_plate_with_hole_grid(; nr=6, ntheta=16, radius=0.25, plate_size=1.0)

Create a structured quadrilateral quarter-plate mesh.
The inner boundary is a quarter circle with radius `radius`; the outer boundary is the square `[0, plate_size] x [0, plate_size]`.
"""
function quarter_plate_with_hole_grid(; nr::Int=6, ntheta::Int=16, radius::Float64=0.25, plate_size::Float64=1.0)
    @assert nr >= 1
    @assert ntheta >= 2
    @assert 0.0 < radius < plate_size

    nodes = Node{2,Float64}[]
    node_id(i, j) = 1 + i * (ntheta + 1) + j

    for i in 0:nr
        s = i / nr
        for j in 0:ntheta
            theta = 0.5 * pi * j / ntheta
            inner = Vec((radius * cos(theta), radius * sin(theta)))
            outer = if theta <= 0.25 * pi
                Vec((plate_size, plate_size * tan(theta)))
            else
                Vec((plate_size / tan(theta), plate_size))
            end
            push!(nodes, Node((1.0 - s) * inner + s * outer))
        end
    end

    cells = Quadrilateral[]
    for i in 0:(nr - 1), j in 0:(ntheta - 1)
        push!(cells, Quadrilateral((
            node_id(i, j),
            node_id(i + 1, j),
            node_id(i + 1, j + 1),
            node_id(i, j + 1),
        )))
    end

    grid = Grid(cells, nodes)
    addfacetset!(grid, "symmetry_x", x -> isapprox(x[2], 0.0; atol=1e-10))
    addfacetset!(grid, "symmetry_y", x -> isapprox(x[1], 0.0; atol=1e-10))
    addfacetset!(grid, "right", x -> isapprox(x[1], plate_size; atol=1e-10))
    addfacetset!(grid, "top", x -> isapprox(x[2], plate_size; atol=1e-10))
    addfacetset!(grid, "hole", x -> isapprox(sqrt(x[1]^2 + x[2]^2), radius; atol=1e-8))
    return grid
end

"""
    run_plate_with_hole(; kwargs...)

Solve a displacement-driven quarter plate with a circular hole using `PlaneStress(NeoHooke(E, nu))`.
Returns a named tuple with the solution, stresses, grid, dof handler, constraint handler, and assembler.
"""
function run_plate_with_hole(;
    nr::Int=6,
    ntheta::Int=16,
    radius::Float64=0.25,
    plate_size::Float64=1.0,
    E::Float64=100.0,
    nu::Float64=0.3,
    displacement::Float64=0.03,
    load_steps::Int=4,
    max_newton::Int=20,
    newton_tol::Float64=1e-8,
)
    grid = quarter_plate_with_hole_grid(; nr, ntheta, radius, plate_size)

    interpolation = Lagrange{RefQuadrilateral,1}()^2
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "symmetry_x"), (x, t) -> [0.0], [2]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "symmetry_y"), (x, t) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [displacement * t], [1]))
    close!(ch)

    material = PlaneStress(NeoHooke(E, nu))
    assembler = create_assembler(material, dh, ch; quadrature_order=2)

    u = zeros(ndofs(dh))
    converged_steps = 0
    for step in 1:load_steps
        t = step / load_steps
        update!(ch, t)
        apply!(u, ch)

        converged = false
        for _ in 1:max_newton
            K, r = stiffness_matrix(assembler, u; dt=1.0 / load_steps)
            apply_zero!(K, r, ch)
            if norm(r) < newton_tol
                converged = true
                break
            end
            u .-= K \ r
        end
        converged || error("Newton iteration did not converge at load step $step")
        update_states!(assembler)
        converged_steps += 1
    end

    stresses = compute_stresses(assembler, u)
    return (
        u=u,
        stresses=stresses,
        grid=grid,
        dh=dh,
        ch=ch,
        assembler=assembler,
        converged_steps=converged_steps,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_plate_with_hole()
    println("Solved plate with hole")
    println("  cells: $(getncells(result.grid))")
    println("  dofs: $(ndofs(result.dh))")
    println("  max displacement: $(maximum(abs, result.u))")
    println("  stress entries: $(length(result.stresses))")
end
