using Test
using Ferrite
using FerriteSolidMechanics
using LinearAlgebra

# Resultants are read per component from the interleaved dof vector: a vector
# field stores component c of every node at dofs c:ncomp:end
resultant(f, comp, ncomp) = sum(@view f[comp:ncomp:end])

function plate_setup(; nx=3, ny=2, Lx=2.0, Ly=1.0, thickness=1.0, order=1)
    grid = generate_grid(Quadrilateral, (nx, ny), Vec(0.0, 0.0), Vec(Lx, Ly))
    ip = Lagrange{RefQuadrilateral,order}()^2
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0], [1, 2]))
    close!(ch)
    assembler = create_assembler(Hooke2D(200.0, 0.3), dh, ch; quadrature_order=2, thickness=thickness)
    return grid, dh, ch, assembler
end

function cube_setup(; n=2)
    grid = generate_grid(Hexahedron, (n, n, n))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0, 0.0], [1, 2, 3]))
    close!(ch)
    assembler = create_assembler(Hooke(200.0, 0.3), dh, ch; quadrature_order=2)
    return grid, dh, ch, assembler
end

# Build a closed handler in one call
function closed_handler(assembler, loads...)
    lh = LoadHandler(assembler)
    for load in loads
        add!(lh, load)
    end
    close!(lh)
    return lh
end

@testset "load resultants match the applied quantity" begin
    Lx, Ly = 2.0, 1.0
    _, dh, _, assembler = plate_setup(; Lx=Lx, Ly=Ly)

    # 2D traction is force per unit length, so a uniform load on the right edge
    # of height Ly integrates to p * Ly
    p = 3.0
    lh = closed_handler(assembler, Traction("right", (x, t) -> Vec(p, 0.0)))
    f = external_forces!(lh, 0.0)
    @test resultant(f, 1, 2) ≈ p * Ly
    @test resultant(f, 2, 2) ≈ 0.0 atol = 1e-12

    # Body force integrates over the area in 2D
    rho_g = 7.0
    lh_body = closed_handler(assembler, BodyForce(x -> Vec(0.0, -rho_g)))
    f_body = external_forces!(lh_body, 0.0)
    @test resultant(f_body, 2, 2) ≈ -rho_g * Lx * Ly
    @test resultant(f_body, 1, 2) ≈ 0.0 atol = 1e-12

    # 3D traction is force per unit area; generate_grid spans [-1,1]^3, face area 4
    _, dh3, _, assembler3 = cube_setup()
    q = 2.5
    lh3 = closed_handler(assembler3, Traction("right", (x, t) -> Vec(q, 0.0, 0.0)))
    f3 = external_forces!(lh3, 0.0)
    @test resultant(f3, 1, 3) ≈ q * 4.0

    # 3D body force integrates over the volume, 2^3
    lh3b = closed_handler(assembler3, BodyForce(x -> Vec(0.0, 0.0, -rho_g)))
    @test resultant(external_forces!(lh3b, 0.0), 3, 3) ≈ -rho_g * 8.0
end

@testset "uniform pressure over a closed surface is self-equilibrated" begin
    _, _, _, assembler = plate_setup()
    lh = LoadHandler(assembler)
    for name in ("left", "right", "top", "bottom")
        add!(lh, Pressure(name, (x, t) -> 2.0))
    end
    close!(lh)
    f = external_forces!(lh, 0.0)
    @test resultant(f, 1, 2) ≈ 0.0 atol = 1e-12
    @test resultant(f, 2, 2) ≈ 0.0 atol = 1e-12
    # Individual nodal values are not zero; only the resultant vanishes
    @test norm(f) > 1.0

    _, _, _, assembler3 = cube_setup()
    lh3 = LoadHandler(assembler3)
    for name in ("left", "right", "top", "bottom", "front", "back")
        add!(lh3, Pressure(name, (x, t) -> 2.0))
    end
    close!(lh3)
    f3 = external_forces!(lh3, 0.0)
    for c in 1:3
        @test resultant(f3, c, 3) ≈ 0.0 atol = 1e-12
    end
end

@testset "Pressure is Traction along the negative normal" begin
    _, _, _, assembler = plate_setup()
    p = 1.75
    lh_p = closed_handler(assembler, Pressure("right", (x, t) -> p * t))
    lh_t = closed_handler(assembler, Traction("right", (x, t, n) -> -p * t * n))
    @test external_forces!(lh_p, 2.0) == external_forces!(lh_t, 2.0)

    # Positive pressure on the right face of a rectangle pushes in -x
    @test resultant(external_forces!(lh_p, 1.0), 1, 2) < 0.0

    # Pressure reaches f(x, t, n) as Traction does, and n is the outward facet
    # normal: on "right" it is (1, 0), so abs(n[1]) scales by one and n[2] by zero
    f_p = copy(external_forces!(lh_p, 2.0))
    lh_n = closed_handler(assembler, Pressure("right", (x, t, n) -> p * t * abs(n[1])))
    @test external_forces!(lh_n, 2.0) ≈ f_p
    lh_tangential = closed_handler(assembler, Pressure("right", (x, t, n) -> p * t * n[2]))
    @test all(iszero, external_forces!(lh_tangential, 2.0))
end

@testset "spatial profiles integrate to the analytic resultant and moment" begin
    Ly = 1.0
    grid, dh, _, assembler = plate_setup(; ny=4, Ly=Ly)

    # Triangular edge load t_x(y) = p * y / Ly: resultant p*Ly/2
    p = 6.0
    lh = closed_handler(assembler, Traction("right", (x, t) -> Vec(p * x[2] / Ly, 0.0)))
    f = external_forces!(lh, 0.0)
    @test resultant(f, 1, 2) ≈ p * Ly / 2

    # dot(f_ext, u) == ∫ t·u dΓ for any exactly represented u, which checks the
    # distribution over the facet rather than only its sum
    probe = zeros(ndofs(dh))
    apply_analytical!(probe, dh, :u, x -> Vec(x[2], 0.0))
    @test dot(f, probe) ≈ p * Ly^2 / 3

    fill!(probe, 0.0)
    apply_analytical!(probe, dh, :u, x -> Vec(1.0, 0.0))
    @test dot(f, probe) ≈ p * Ly / 2

    # Quadratic in space and sinusoidal in time evaluates at the requested t
    lh_mixed = closed_handler(assembler, Traction("right", (x, t) -> Vec(p * (x[2] / Ly)^2 * sin(t), 0.0)))
    @test resultant(external_forces!(lh_mixed, pi / 2), 1, 2) ≈ p * Ly / 3
    @test resultant(external_forces!(lh_mixed, 0.0), 1, 2) ≈ 0.0 atol = 1e-12
end

@testset "thickness scales facet and body loads like ke and re" begin
    _, _, _, assembler1 = plate_setup(; thickness=1.0)
    _, _, _, assembler2 = plate_setup(; thickness=2.5)

    lh1 = closed_handler(assembler1, Traction("right", (x, t) -> Vec(3.0, 0.0)), BodyForce(x -> Vec(0.0, -1.0)))
    lh2 = closed_handler(assembler2, Traction("right", (x, t) -> Vec(3.0, 0.0)), BodyForce(x -> Vec(0.0, -1.0)))

    f1 = copy(external_forces!(lh1, 0.0))
    f2 = external_forces!(lh2, 0.0)
    @test f2 ≈ 2.5 .* f1

    # A LoadHandler built from a DofHandler defaults to thickness 1.0
    _, dh, _, _ = plate_setup(; thickness=2.5)
    lh_dh = LoadHandler(dh)
    add!(lh_dh, Traction("right", (x, t) -> Vec(3.0, 0.0)))
    add!(lh_dh, BodyForce(x -> Vec(0.0, -1.0)))
    close!(lh_dh)
    @test external_forces!(lh_dh, 0.0) ≈ f1
end

@testset "nodal forces lump correctly in both distribute modes" begin
    grid, _, _, assembler = plate_setup(; nx=3, ny=2)
    addnodeset!(grid, "right_edge", x -> x[1] ≈ 2.0)
    nnodes_set = length(getnodeset(grid, "right_edge"))
    @test nnodes_set > 1

    F = -5.0
    lh_each = closed_handler(assembler, NodalForce("right_edge", x -> Vec(0.0, F)))
    @test resultant(external_forces!(lh_each, 0.0), 2, 2) ≈ F * nnodes_set

    lh_total = closed_handler(assembler, NodalForce("right_edge", x -> Vec(0.0, F); distribute=true))
    f_total = external_forces!(lh_total, 0.0)
    @test resultant(f_total, 2, 2) ≈ F
    # Equal splitting: exactly the loaded nodes are nonzero and share alike
    shares = filter(!iszero, f_total[2:2:end])
    @test length(shares) == nnodes_set
    @test all(s -> s ≈ F / nnodes_set, shares)

    # Time dependence reaches nodal forces too
    lh_t = closed_handler(assembler, NodalForce("right_edge", (x, t) -> Vec(0.0, F * t); distribute=true))
    @test resultant(external_forces!(lh_t, 3.0), 2, 2) ≈ 3.0 * F
end

@testset "load function call forms are all reachable" begin
    _, _, _, assembler = plate_setup()
    value = Vec(2.0, -1.0)

    lh_x = closed_handler(assembler, Traction("right", x -> value))
    lh_xt = closed_handler(assembler, Traction("right", (x, t) -> value))
    f_x = copy(external_forces!(lh_x, 7.0))
    @test f_x == external_forces!(lh_xt, 7.0)

    # f(x, t, n) is preferred when available
    lh_xtn = closed_handler(assembler, Traction("right", (x, t, n) -> 2.0 * n))
    @test resultant(external_forces!(lh_xtn, 0.0), 1, 2) ≈ 2.0 * 1.0

    # A plain vector return is accepted and converted
    lh_vec = closed_handler(assembler, Traction("right", x -> [2.0, -1.0]))
    @test external_forces!(lh_vec, 0.0) ≈ f_x

    # f(x) ignores t, f(x, t) does not
    lh_t = closed_handler(assembler, Traction("right", (x, t) -> value * t))
    @test external_forces!(lh_t, 2.0) ≈ 2.0 .* f_x
end

@testset "close! prepares only new entries and can be called again" begin
    _, _, _, assembler = plate_setup()
    traction = Traction("right", (x, t) -> Vec(1.0, 0.0))
    body = BodyForce(x -> Vec(0.0, -2.0))

    lh = LoadHandler(assembler)
    add!(lh, traction)
    close!(lh)
    f_first = copy(external_forces!(lh, 1.0))

    add!(lh, body)
    # Adding without closing is caught rather than silently ignored
    @test_throws ArgumentError external_forces!(lh, 1.0)
    close!(lh)
    f_second = copy(external_forces!(lh, 1.0))

    lh_upfront = closed_handler(assembler, traction, body)
    @test f_second == external_forces!(lh_upfront, 1.0)
    @test f_second != f_first

    # close! on an already closed handler is a no-op
    close!(lh)
    close!(lh)
    @test external_forces!(lh, 1.0) == f_second
end

@testset "loads can change over a simulation without reopening" begin
    _, _, _, assembler = plate_setup()

    # Closing over solver state, not just t
    stage = Ref(1)
    lh = closed_handler(assembler, Traction("right", (x, t) -> stage[] == 1 ? Vec(1.0, 0.0) : zero(Vec{2})))
    @test resultant(external_forces!(lh, 0.0), 1, 2) ≈ 1.0
    stage[] = 2
    @test resultant(external_forces!(lh, 0.0), 1, 2) ≈ 0.0 atol = 1e-12

    # Ramp, hold and release expressed in t alone
    lh_profile = closed_handler(assembler, Traction("right", (x, t) -> t < 2.0 ? Vec(t, 0.0) : zero(Vec{2})))
    @test resultant(external_forces!(lh_profile, 1.0), 1, 2) ≈ 1.0
    @test resultant(external_forces!(lh_profile, 3.0), 1, 2) ≈ 0.0 atol = 1e-12
end

@testset "invalid loads are rejected at close! with a named entry" begin
    grid, dh, _, assembler = plate_setup()

    unknown_set = LoadHandler(assembler)
    add!(unknown_set, Traction("does_not_exist", x -> Vec(1.0, 0.0)))
    @test_throws ArgumentError close!(unknown_set)

    empty_set = LoadHandler(assembler)
    addfacetset!(grid, "empty", x -> false)
    add!(empty_set, Traction("empty", x -> Vec(1.0, 0.0)))
    @test_throws ArgumentError close!(empty_set)

    wrong_length = LoadHandler(assembler)
    add!(wrong_length, Traction("right", x -> Vec(1.0, 0.0, 0.0)))
    @test_throws ArgumentError close!(wrong_length)

    scalar_traction = LoadHandler(assembler)
    add!(scalar_traction, Traction("right", x -> 1.0))
    @test_throws ArgumentError close!(scalar_traction)

    vector_pressure = LoadHandler(assembler)
    add!(vector_pressure, Pressure("right", x -> Vec(1.0, 0.0)))
    @test_throws ArgumentError close!(vector_pressure)

    bad_arity = LoadHandler(assembler)
    add!(bad_arity, Traction("right", () -> Vec(1.0, 0.0)))
    @test_throws ArgumentError close!(bad_arity)

    # The message names which entry failed
    err = try
        close!(LoadHandler(assembler))
        lh = LoadHandler(assembler)
        add!(lh, Traction("right", x -> Vec(1.0, 0.0)))
        add!(lh, Traction("does_not_exist", x -> Vec(1.0, 0.0)))
        close!(lh)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("load 2", err.msg)
end

@testset "nodes without dofs are rejected rather than silently dropped" begin
    grid = generate_grid(Quadrilateral, (2, 2), Vec(0.0, 0.0), Vec(2.0, 2.0))
    addcellset!(grid, "active", Set(1:2))
    dh = DofHandler(grid)
    sdh = SubDofHandler(dh, getcellset(grid, "active"))
    add!(sdh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)
    assembler = create_assembler(Dict("active" => Hooke2D(200.0, 0.3)), dh, ch; quadrature_order=2)

    addnodeset!(grid, "all_nodes", x -> true)
    lh = LoadHandler(assembler)
    add!(lh, NodalForce("all_nodes", x -> Vec(0.0, -1.0)))
    @test_throws ArgumentError close!(lh)

    # Facets on cells outside the DofHandler are rejected the same way
    addfacetset!(grid, "everywhere", x -> true)
    lh_facet = LoadHandler(assembler)
    add!(lh_facet, Traction("everywhere", x -> Vec(1.0, 0.0)))
    @test_throws ArgumentError close!(lh_facet)
end

@testset "sets may be passed as objects instead of names" begin
    grid, _, _, assembler = plate_setup()
    addnodeset!(grid, "right_edge", x -> x[1] ≈ 2.0)
    addcellset!(grid, "all", Set(1:getncells(grid)))

    by_name = closed_handler(assembler,
        Traction("right", x -> Vec(1.0, 0.0)),
        NodalForce("right_edge", x -> Vec(0.0, -1.0)),
        BodyForce("all", x -> Vec(0.0, -1.0)))
    by_object = closed_handler(assembler,
        Traction(getfacetset(grid, "right"), x -> Vec(1.0, 0.0)),
        NodalForce(getnodeset(grid, "right_edge"), x -> Vec(0.0, -1.0)),
        BodyForce(getcellset(grid, "all"), x -> Vec(0.0, -1.0)))
    @test external_forces!(by_name, 0.0) == external_forces!(by_object, 0.0)

    # BodyForce without a set covers every cell of the DofHandler
    all_cells = closed_handler(assembler, BodyForce(x -> Vec(0.0, -1.0)))
    set_cells = closed_handler(assembler, BodyForce("all", x -> Vec(0.0, -1.0)))
    @test external_forces!(all_cells, 0.0) == external_forces!(set_cells, 0.0)
end

@testset "loads reproduce a hand-written facet integration loop" begin
    # The reference is the loop this feature replaces in the four-point bending
    # example: FacetValues, quadrature weights, dot with the shape value
    grid = generate_grid(Quadrilateral, (10, 2), Vec(0.0, 0.0), Vec(10.0, 1.0))
    interpolation = Lagrange{RefQuadrilateral,1}()^2
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0], [1, 2]))
    close!(ch)

    pressure = 0.4
    facetvalues = FacetValues(Float64, FacetQuadratureRule{RefQuadrilateral}(2), interpolation)
    reference = zeros(ndofs(dh))
    traction = Vec(0.0, -pressure)
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for facet in FacetIterator(dh, getfacetset(grid, "top"))
        reinit!(facetvalues, facet)
        fill!(fe_ext, 0.0)
        for qp in 1:getnquadpoints(facetvalues)
            dGamma = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                fe_ext[i] += dot(traction, shape_value(facetvalues, qp, i)) * dGamma
            end
        end
        assemble!(reference, celldofs(facet), fe_ext)
    end

    assembler = create_assembler(Hooke2D(200.0, 0.3), dh, ch; quadrature_order=2)
    lh = closed_handler(assembler, Traction("top", (x, t) -> Vec(0.0, -pressure)))
    @test external_forces!(lh, 0.0) == reference
end

@testset "traction-loaded bar reproduces the analytic elongation" begin
    # Uniaxial bar under end traction p: u_x(L) = p * L / E, with the lateral
    # faces free and only enough constraints to remove rigid body motion
    L, H = 4.0, 1.0
    E, nu = 210.0, 0.3
    p = 1.5

    grid = generate_grid(Quadrilateral, (8, 2), Vec(0.0, 0.0), Vec(L, H))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0], [1]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> [0.0], [2]))
    close!(ch)

    # Plane stress, so the 1D relation u = pL/E holds exactly
    assembler = create_assembler(Hooke2D(E, nu; plane_stress=true), dh, ch; quadrature_order=2)
    lh = closed_handler(assembler, Traction("right", (x, t) -> Vec(p * t, 0.0)))

    u = zeros(ndofs(dh))
    residual = zeros(ndofs(dh))
    f_ext = external_forces!(lh, 1.0)

    apply!(u, ch)
    converged = false
    for _ in 1:10
        K, f_int = stiffness_matrix(assembler, u; dt=0.0)
        residual .= f_int .- f_ext
        apply_zero!(K, residual, ch)
        if norm(residual) <= 1e-10 * max(norm(f_ext), 1.0)
            converged = true
            break
        end
        u .-= K \ residual
    end
    @test converged

    ux_tip = maximum(v[1] for v in evaluate_at_grid_nodes(dh, u, :u))
    @test ux_tip ≈ p * L / E rtol = 1e-10

    # Halving the load halves the force. The handler owns the returned vector,
    # so the first result has to be copied before the second call overwrites it.
    f_half = copy(external_forces!(lh, 0.5))
    @test f_half ≈ 0.5 .* external_forces!(lh, 1.0)
end

@testset "external_forces! reuses its buffer and reports it" begin
    _, _, _, assembler = plate_setup()
    lh = closed_handler(assembler, Traction("right", (x, t) -> Vec(1.0, 0.0) * t))

    f1 = external_forces!(lh, 1.0)
    f2 = external_forces!(lh, 2.0)
    @test f1 === f2 # documented: the vector is owned by the handler

    # Default t is zero
    lh_static = closed_handler(assembler, Traction("right", x -> Vec(1.0, 0.0)))
    @test external_forces!(lh_static) == external_forces!(lh_static, 0.0)

    @test occursin("LoadHandler", sprint(show, lh))
    @test occursin("Traction", sprint(show, lh))
    open_handler = LoadHandler(assembler)
    add!(open_handler, Traction("right", x -> Vec(1.0, 0.0)))
    @test occursin("not yet closed", sprint(show, open_handler))
end

@testset "higher-order interpolations integrate consistently" begin
    # Second-order elements distribute a uniform traction unequally over the
    # facet nodes, but the resultant is unchanged
    Ly = 1.0
    _, _, _, assembler = plate_setup(; order=2, Ly=Ly)
    p = 3.0
    lh = closed_handler(assembler, Traction("right", (x, t) -> Vec(p, 0.0)))
    f = external_forces!(lh, 0.0)
    @test resultant(f, 1, 2) ≈ p * Ly

    nodal = filter(!iszero, f[1:2:end])
    @test length(unique(round.(nodal; digits=12))) > 1
end
