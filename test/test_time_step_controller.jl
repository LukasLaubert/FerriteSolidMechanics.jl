using Test
using FerriteSolidMechanics

@testset "TimeStepController defaults" begin
    controller = TimeStepController()

    @test isnan(controller.dt_min)
    @test isnan(controller.dt_max)
    @test controller.shrink == 0.5
    @test controller.grow == 2.0
    @test controller.max_rejections == typemax(Int)
    @test controller.rejections == 0
    @test sprint(show, controller) == "TimeStepController(dt_min=NaN, dt_max=NaN, shrink=0.5, grow=2.0, max_rejections=$(typemax(Int)), rejections=0)"
end

@testset "TimeStepController accept/reject updates" begin
    controller = TimeStepController(; dt_min=1.0e-3, dt_max=0.25, shrink=0.5, grow=2.0, max_rejections=2)

    @test reject_step!(controller, 0.1) == 0.05
    @test controller.rejections == 1
    @test reject_step!(controller, 0.05) == 0.025
    @test controller.rejections == 2

    @test accept_step!(controller, 0.05) == 0.1
    @test controller.rejections == 0

    @test reject_step!(controller, 0.05) == 0.025
    @test reset_controller!(controller) === controller
    @test controller.rejections == 0

    @test accept_step!(controller, 0.2) == 0.25
    @test accept_step!(controller, 0.25) == 0.25
    @test accept_step!(controller, 2.0) == 0.25
end

@testset "TimeStepController disabled bounds" begin
    controller = TimeStepController(; dt_min=NaN, dt_max=NaN, shrink=0.25, grow=4.0, max_rejections=2)

    @test reject_step!(controller, 1.0e-6) == 2.5e-7
    @test accept_step!(controller, 1.0) == 4.0
end

@testset "TimeStepController validation and exhaustion" begin
    @test_throws ArgumentError TimeStepController(; dt_min=0.0)
    @test_throws ArgumentError TimeStepController(; dt_max=-1.0)
    @test_throws ArgumentError TimeStepController(; dt_min=0.2, dt_max=0.1)
    @test_throws ArgumentError TimeStepController(; shrink=1.0)
    @test_throws ArgumentError TimeStepController(; grow=0.5)
    @test_throws ArgumentError TimeStepController(; max_rejections=-1)

    dt_min = 0.1
    controller = TimeStepController(; dt_min)
    @test reject_step!(controller, 0.2) == dt_min
    @test controller.rejections == 1
    @test_throws TimeStepControllerExhausted reject_step!(controller, 0.1)
    @test controller.rejections == 1

    controller = TimeStepController(; max_rejections=1)
    @test reject_step!(controller, 1.0) == 0.5
    @test_throws TimeStepControllerExhausted reject_step!(controller, 0.5)
    @test controller.rejections == 1

    controller = TimeStepController(; max_rejections=0)
    @test_throws TimeStepControllerExhausted reject_step!(controller, 1.0)
    @test controller.rejections == 0

    @test_throws ArgumentError accept_step!(TimeStepController(), 0.0)
    @test_throws ArgumentError reject_step!(TimeStepController(), Inf)
    @test_throws ArgumentError accept_step!(TimeStepController(), NaN)
    @test_throws ArgumentError reject_step!(TimeStepController(), -1.0)
end
