using Test

include(joinpath(@__DIR__, "..", "examples", "plate_with_hole_planestress.jl"))
include(joinpath(@__DIR__, "..", "examples", "cantilever_beam_dma.jl"))
include(joinpath(@__DIR__, "..", "examples", "adaptive_time_stepping.jl"))

@testset "Plate-with-hole example integration test" begin
    result = run_plate_with_hole(; nr=2, ntheta=4, load_steps=1, displacement=0.005)
    @test result.converged_steps == 1
    @test maximum(abs, result.u) > 0.0
    @test all(isfinite, result.u)
    @test size(result.stresses, 2) == Ferrite.getncells(result.grid)
end

@testset "Cantilever-beam DMA example integration test" begin
    result = run_cantilever_beam_dma(; nx=3, ny=1, nz=1, nsteps=4, total_time=8.0, amplitude=0.05)
    @test result.converged_steps == 4
    @test maximum(abs, result.u) > 0.0
    @test all(isfinite, result.u)
    @test size(result.stresses, 2) == Ferrite.getncells(result.grid)
    @test 0.0 <= phase_lag_degrees(result) <= 180.0
end

@testset "Adaptive time stepping example integration test" begin
    result = run_adaptive_time_stepping()
    @test result.accepted_steps > 0
    @test result.rejected_steps > 0
    @test result.dt_decreases == result.rejected_steps
    @test result.dt_increases >= 0
    @test minimum(result.history_dt) < 0.25
    @test maximum(abs, result.u) > 0.0
    @test all(isfinite, result.u)
    @test size(result.stresses, 2) == Ferrite.getncells(result.grid)
    # The adaptive driver must reach t_end exactly.
    @test isempty(result.history_t) || isapprox(result.history_t[end], 1.0; atol=1e-9)
end
