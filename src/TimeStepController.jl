"""
    TimeStepControllerExhausted

Exception thrown by [`reject_step!`](@ref) when the controller has no smaller time step to offer, because the proposed step would violate `dt_min` or the consecutive rejection budget.
"""
struct TimeStepControllerExhausted <: Exception
    reason::Symbol
    dt::Float64
    next_dt::Float64
    dt_min::Float64
    rejections::Int
    max_rejections::Int
end

function Base.showerror(io::IO, err::TimeStepControllerExhausted)
    if err.reason === :dt_min
        print(io, "adaptive time step controller exhausted: rejected step would reduce dt from $(err.dt) to $(err.next_dt) below dt_min=$(err.dt_min)")
    elseif err.reason === :max_rejections
        print(io, "adaptive time step controller exhausted: maximum consecutive rejected time steps exceeded (rejections=$(err.rejections), max_rejections=$(err.max_rejections), dt=$(err.dt))")
    else
        print(io, "adaptive time step controller exhausted")
    end
end

"""
    TimeStepController(; dt_min=NaN, dt_max=NaN, shrink=0.5, grow=2.0,
                       max_rejections=typemax(Int))

Adaptive time step controller for user-owned load and time loops.

The controller runs no assembly, Newton iterations, rollback, or MPI communication.
It only updates scalar time step sizes after the caller has accepted or rejected a trial step, which makes it usable with any material model and with serial, threaded, or MPI assembly.

`dt_min` and `dt_max` are optional bounds, and `NaN` disables the corresponding lower or upper bound.
`shrink` must satisfy `0 < shrink < 1`.
`grow` must be finite and satisfy `grow >= 1`.
Both factors are fixed when the controller is constructed; [`accept_step!`](@ref) and [`reject_step!`](@ref) do not adapt them.
`max_rejections = N` allows `N` consecutive rejected steps since the last accepted step and throws on rejection `N + 1`.
The default leaves that count practically unbounded.

`rejections` counts the rejected steps since the last accepted one.
[`reject_step!`](@ref) increments it, [`accept_step!`](@ref) resets it to zero, and [`reset_controller!`](@ref) clears it without returning a new step size.
It is the only field meant to change after construction; leave `dt_min`, `dt_max`, `shrink`, `grow` and `max_rejections` at the values given to the constructor.

Use one `TimeStepController` per Newton loop.
The mutating functions assume a single caller and are not safe to call from several tasks at once.
"""
mutable struct TimeStepController
    dt_min::Float64
    dt_max::Float64
    shrink::Float64
    grow::Float64
    max_rejections::Int
    rejections::Int
end

function TimeStepController(; dt_min=NaN, dt_max=NaN, shrink=0.5, grow=2.0,
                            max_rejections::Integer=typemax(Int))
    dt_min = Float64(dt_min)
    dt_max = Float64(dt_max)
    shrink = Float64(shrink)
    grow = Float64(grow)
    _validate_dt_bound(dt_min, :dt_min)
    _validate_dt_bound(dt_max, :dt_max)
    if _has_dt_bound(dt_min) && _has_dt_bound(dt_max) && dt_max < dt_min
        throw(ArgumentError("dt_max must be >= dt_min when both bounds are enabled (dt_min=$dt_min, dt_max=$dt_max)"))
    end
    0.0 < shrink < 1.0 || throw(ArgumentError("shrink must satisfy 0 < shrink < 1"))
    grow >= 1.0 && isfinite(grow) || throw(ArgumentError("grow must be finite and >= 1"))
    max_rejections >= 0 || throw(ArgumentError("max_rejections must be >= 0"))
    return TimeStepController(dt_min, dt_max, shrink, grow, Int(max_rejections), 0)
end

function Base.show(io::IO, controller::TimeStepController)
    print(io, "TimeStepController(",
          "dt_min=", controller.dt_min,
          ", dt_max=", controller.dt_max,
          ", shrink=", controller.shrink,
          ", grow=", controller.grow,
          ", max_rejections=", controller.max_rejections,
          ", rejections=", controller.rejections,
          ")")
end

"""
    reset_controller!(controller)

Reset the controller's consecutive rejection counter to zero and return the
controller.
"""
function reset_controller!(controller::TimeStepController)
    controller.rejections = 0
    return controller
end

"""
    accept_step!(controller, dt)

Return the next trial time step after an accepted step.

`accept_step!` resets the controller's consecutive rejection counter, multiplies `dt` by `controller.grow`, and clamps the result to `controller.dt_max` when that upper bound is enabled.
`dt_max = NaN` disables the upper clamp.
"""
function accept_step!(controller::TimeStepController, dt::Real)
    dt = _validate_dt(dt)
    reset_controller!(controller)
    next_dt = dt * controller.grow
    if _has_dt_bound(controller.dt_max)
        next_dt = min(next_dt, controller.dt_max)
    end
    return next_dt
end

"""
    reject_step!(controller, dt)

Return the next trial time step after a rejected step.

`reject_step!` computes `dt * controller.shrink`, checks the lower bound and the rejection budget, then increments the controller's consecutive rejection counter on success.
It throws when the new step would fall below `controller.dt_min` and that lower bound is enabled.
It does not clamp rejected steps to `dt_min`.
`dt_min = NaN` disables the lower-bound check.
The controller state is left unchanged when `reject_step!` throws.
"""
function reject_step!(controller::TimeStepController, dt::Real)
    dt = _validate_dt(dt)
    next_dt = dt * controller.shrink
    if _has_dt_bound(controller.dt_min) && next_dt < controller.dt_min
        throw(TimeStepControllerExhausted(:dt_min, dt, next_dt, controller.dt_min, controller.rejections, controller.max_rejections))
    end
    next_rejections = controller.rejections + 1
    if next_rejections > controller.max_rejections
        throw(TimeStepControllerExhausted(:max_rejections, dt, next_dt, controller.dt_min, next_rejections, controller.max_rejections))
    end
    controller.rejections = next_rejections
    return next_dt
end

_has_dt_bound(dt::Float64) = !isnan(dt)

function _validate_dt_bound(dt::Float64, name::Symbol)
    if _has_dt_bound(dt) && !(isfinite(dt) && dt > 0.0)
        throw(ArgumentError("$name must be positive and finite, or NaN to disable the bound (got $dt)"))
    end
    return nothing
end

function _validate_dt(dt::Real)
    dt = Float64(dt)
    isfinite(dt) && dt > 0.0 || throw(ArgumentError("dt must be positive and finite"))
    return dt
end