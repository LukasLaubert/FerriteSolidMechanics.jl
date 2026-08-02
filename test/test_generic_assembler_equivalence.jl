using Test
using Ferrite
using FerriteSolidMechanics
using LinearAlgebra
using Tensors

struct DTAwareMaterial <: AbstractMaterial end

function FerriteSolidMechanics._assemble_element!(ke, re, states, ::DTAwareMaterial, cellvalues, alphavalues, u, dt)
    if re !== nothing
        fill!(re, dt)
    end
    if ke !== nothing
        fill!(ke, 0.0)
    end
    return nothing
end

FerriteSolidMechanics._compute_stress_qp(::DTAwareMaterial, cellvalues, alphavalues, qp, u_local, state, dt=0.0) =
    zero(Tensor{2,2})

struct DenseBufferProbeMaterial <: AbstractMaterial
    marker::Float64
end

FerriteSolidMechanics.is_linear(::DenseBufferProbeMaterial) = false
FerriteSolidMechanics.create_state(::DenseBufferProbeMaterial) = NoState()

function _fill_probe_element!(ke, re, marker, cellvalues)
    expected_size = (getnbasefunctions(cellvalues), getnbasefunctions(cellvalues))
    size(ke) == expected_size || error(:probe_matrix_size)
    size(re) == (expected_size[1],) || error(:probe_residual_size)
    fill!(ke, 0.0)
    fill!(re, marker)
    for i in axes(ke, 1)
        ke[i, i] = marker
    end
    return nothing
end

# Probe: element routines must receive a dense buffer sized to their own cell
function FerriteSolidMechanics._assemble_element!(ke, re, states, mp::DenseBufferProbeMaterial, cellvalues, alphavalues, u, dt)
    ke isa Matrix || error(:expected_matrix_ke)
    re isa Vector || error(:expected_vector_re)
    return _fill_probe_element!(ke, re, mp.marker, cellvalues)
end

FerriteSolidMechanics._compute_stress_qp(::DenseBufferProbeMaterial, cellvalues, alphavalues, qp, u_local, state, dt=0.0) =
    zero(Tensor{2,2})

struct TestAlphaSource
    scale::Float64
    reinit_cells::Vector{Int}
end

mutable struct TestAlphaValues
    source::TestAlphaSource
    cellid::Int
end

FerriteSolidMechanics.create_alpha_values(ah::TestAlphaSource, cellvalues) = TestAlphaValues(ah, 0)

function Ferrite.reinit!(av::TestAlphaValues, cellid::Int)
    av.cellid = cellid
    push!(av.source.reinit_cells, cellid)
    return av
end

FerriteSolidMechanics.alpha_value(av::TestAlphaValues, qp::Int) = av.source.scale

function assemble_hooke2d_reference(dh, ch, material; quadrature_order=2, thickness=1.0)
    grid = dh.grid
    ip = Lagrange{RefQuadrilateral,1}()^2
    qr = QuadratureRule{RefQuadrilateral}(quadrature_order)
    cellvalues = CellValues(qr, ip)
    K = allocate_matrix(dh, ch)
    assembler = start_assemble(K)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
        for qp in 1:getnquadpoints(cellvalues)
            dOmega = getdetJdV(cellvalues, qp)
            for i in 1:getnbasefunctions(cellvalues), j in 1:getnbasefunctions(cellvalues)
                grad_i = shape_symmetric_gradient(cellvalues, qp, i)
                grad_j = shape_symmetric_gradient(cellvalues, qp, j)
                ke[i, j] += (grad_i ⊡ material.C ⊡ grad_j) * dOmega * thickness
            end
        end
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end

@testset "Generic assembler matches manual Ferrite loop" begin
    grid = generate_grid(Quadrilateral, (2, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    material = Hooke2D(100.0, 0.3; plane_stress=true)
    generic = create_assembler(material, dh, ch; quadrature_order=2)
    K_generic, r_generic = stiffness_matrix(generic, collect(range(0.0, 0.01; length=ndofs(dh))))

    K_reference = assemble_hooke2d_reference(dh, ch, material; quadrature_order=2)
    u = collect(range(0.0, 0.01; length=ndofs(dh)))

    result = try_stiffness_matrix(generic, u)
    @test result.converged
    @test result.K !== nothing
    @test result.r !== nothing
    @test result.error === nothing

    thick = 2.5
    generic_thick = create_assembler(material, dh, ch; quadrature_order=2, thickness=thick)
    K_thick, r_thick = stiffness_matrix(generic_thick, u)
    K_reference_thick = assemble_hooke2d_reference(dh, ch, material; quadrature_order=2, thickness=thick)
    @test isapprox(Matrix(K_thick), Matrix(K_reference_thick); atol=1e-10, rtol=1e-10)
    @test isapprox(r_thick, K_reference_thick * u; atol=1e-10, rtol=1e-10)
    @test compute_stresses(generic_thick, u) ≈ compute_stresses(generic, u)

    @test Matrix(K_generic) ≈ Matrix(K_reference) atol=1e-10 rtol=1e-10
    @test r_generic ≈ K_reference * u atol=1e-10 rtol=1e-10
end

@testset "AlphaValues hook scales assembly and stress output" begin
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    material = Hooke2D(100.0, 0.3; plane_stress=true)
    u = collect(range(0.0, 0.01; length=ndofs(dh)))

    unscaled = create_assembler(material, dh, ch; quadrature_order=2)
    K_unscaled, r_unscaled = stiffness_matrix(unscaled, u)
    stresses_unscaled = copy(compute_stresses(unscaled, u))

    ah = TestAlphaSource(2.0, Int[])
    scaled = create_assembler(material, dh, ch; ah=ah, quadrature_order=2)
    K_scaled, r_scaled = stiffness_matrix(scaled, u)
    stresses_scaled = copy(compute_stresses(scaled, u))

    @test Matrix(K_scaled) ≈ 2.0 .* Matrix(K_unscaled)
    @test r_scaled ≈ 2.0 .* r_unscaled
    @test stresses_scaled ≈ 2.0 .* stresses_unscaled
    @test !isempty(ah.reinit_cells)
    @test all(==(1), ah.reinit_cells)
end

@testset "compute_forces forwards dt" begin
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    fem = create_assembler(DTAwareMaterial(), dh, ch; quadrature_order=2)
    u = zeros(ndofs(dh))

    @test all(==(2.5), compute_forces(fem, u; dt=2.5))
end

@testset "material cellset validation" begin
    grid = generate_grid(Quadrilateral, (2, 1))
    addcellset!(grid, "left", [1])
    addcellset!(grid, "right", [2])
    addcellset!(grid, "all", [1, 2])

    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    @test_throws ErrorException create_assembler(Dict("left" => Hooke2D(100.0, 0.3)), dh, ch)
    @test_throws ErrorException create_assembler(Dict(
        "left" => Hooke2D(100.0, 0.3),
        "all" => Hooke2D(100.0, 0.3),
    ), dh, ch)

    fem = create_assembler(Dict(
        "left" => Hooke2D(100.0, 0.3),
        "right" => Hooke2D(50.0, 0.3),
    ), dh, ch)
    @test count(!iszero, fem.cell_to_mat_idx) == getncells(grid)

    inactive_grid = generate_grid(Quadrilateral, (3, 1))
    addcellset!(inactive_grid, "left", [1, 3])
    addcellset!(inactive_grid, "right", [2])

    inactive_dh = DofHandler(inactive_grid)
    inactive_sdh = SubDofHandler(inactive_dh, Set([1, 2]))
    add!(inactive_sdh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(inactive_dh)

    inactive_ch = ConstraintHandler(inactive_dh)
    close!(inactive_ch)

    inactive_fem = create_assembler(Dict(
        "left" => Hooke2D(100.0, 0.3),
        "right" => Hooke2D(50.0, 0.3),
    ), inactive_dh, inactive_ch)
    @test inactive_fem.cell_to_mat_idx[1] != 0
    @test inactive_fem.cell_to_mat_idx[2] != 0
    @test inactive_fem.cell_to_mat_idx[3] == 0
    @test count(!iszero, inactive_fem.cell_to_mat_idx) == 2
end

@testset "mixed linear and nonlinear material dictionary" begin
    grid = generate_grid(Hexahedron, (2, 1, 1))
    addcellset!(grid, "left", [1])
    addcellset!(grid, "right", [2])

    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron,1}()^3)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    fem = create_assembler(Dict(
        "left" => Hooke(100.0, 0.3),
        "right" => NeoHooke(50.0, 0.3),
    ), dh, ch; quadrature_order=1)
    u = collect(range(0.0, 1.0e-4; length=ndofs(dh)))
    K, r = stiffness_matrix(fem, u; dt=0.1)

    @test fem.linear_cells == [1]
    @test fem.nonlinear_cells == [2]
    @test all(isfinite, K.nzval)
    @test all(isfinite, r)
end

@testset "element routines get dense exact-size buffers across mixed batches and SDHs" begin
    # Two materials sharing one SubDofHandler
    grid = generate_grid(Quadrilateral, (2, 1))
    addcellset!(grid, string(:left), [1])
    addcellset!(grid, string(:right), [2])

    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    fem = create_assembler(Dict(
        string(:left) => DenseBufferProbeMaterial(1.0),
        string(:right) => DenseBufferProbeMaterial(2.0),
    ), dh, ch; quadrature_order=2)

    si = fem._sdh_cell_to_sdh[1]
    @test si == fem._sdh_cell_to_sdh[2]
    @test length(fem._sdh_owned_nonlinear[si]) == 2

    # The probe errors on a wrong buffer, so a clean assembly is the assertion
    K, r = stiffness_matrix(fem, zeros(ndofs(dh)); dt=0.0)
    @test all(isfinite, K.nzval)
    @test all(isfinite, r)

    # Two SubDofHandlers with different ndofs_per_cell
    grid = generate_grid(Quadrilateral, (2, 1))
    addcellset!(grid, string(:left), [1])
    addcellset!(grid, string(:right), [2])

    dh = DofHandler(grid)
    sdh_left = SubDofHandler(dh, getcellset(grid, string(:left)))
    add!(sdh_left, :u, Lagrange{RefQuadrilateral,1}()^2)
    sdh_right = SubDofHandler(dh, getcellset(grid, string(:right)))
    add!(sdh_right, :u, Lagrange{RefQuadrilateral,2}()^2)
    close!(dh)

    ch = ConstraintHandler(dh)
    close!(ch)

    fem = create_assembler(Dict(
        string(:left) => DenseBufferProbeMaterial(1.0),
        string(:right) => DenseBufferProbeMaterial(2.0),
    ), dh, ch; quadrature_order=[2, 3])

    left_si = fem._sdh_cell_to_sdh[1]
    right_si = fem._sdh_cell_to_sdh[2]
    @test left_si != right_si
    @test fem._sdh_ndofs[left_si] != fem._sdh_ndofs[right_si]
    @test length(fem._sdh_owned_nonlinear[left_si]) == 1
    @test length(fem._sdh_owned_nonlinear[right_si]) == 1
    for ws in fem._workspaces
        @test size(ws.kes[left_si]) == (fem._sdh_ndofs[left_si], fem._sdh_ndofs[left_si])
        @test size(ws.kes[right_si]) == (fem._sdh_ndofs[right_si], fem._sdh_ndofs[right_si])
        @test length(ws.res[left_si]) == fem._sdh_ndofs[left_si]
        @test length(ws.res[right_si]) == fem._sdh_ndofs[right_si]
    end

    K, r = stiffness_matrix(fem, zeros(ndofs(dh)); dt=0.0)
    @test all(isfinite, K.nzval)
    @test all(isfinite, r)
end