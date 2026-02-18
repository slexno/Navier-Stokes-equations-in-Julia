"""
Utilities for a 2D staggered (MAC) grid discretization.

Grid arrangement:
- pressure `p`: `(Nx, Ny)` cell centers
- x-velocity `u`: `(Nx + 1, Ny)` x-normal faces
- y-velocity `v`: `(Nx, Ny + 1)` y-normal faces
"""



Nx, Ny = 3, 3
dx, dy = 0.5, 0.5

p = zeros(Nx, Ny)
u = zeros(Nx + 1, Ny)
v = zeros(Nx, Ny + 1)

nu = 1e-3



function _check_sizes(u, v, Nx::Int, Ny::Int)
    size(u) == (Nx + 1, Ny) || throw(ArgumentError("u must have size ($(Nx+1), $Ny)"))
    size(v) == (Nx, Ny + 1) || throw(ArgumentError("v must have size ($Nx, $(Ny+1))"))
end


"""Apply no-slip wall conditions on all domain borders (all boundary face velocities set to zero)."""
function apply_wall_boundaries!(u::AbstractMatrix, v::AbstractMatrix)
    u[1, :] .= 0.0
    u[end, :] .= 0.0
    u[:, 1] .= 0.0
    u[:, end] .= 0.0

    v[:, 1] .= 0.0
    v[:, end] .= 0.0
    v[1, :] .= 0.0
    v[end, :] .= 0.0
    return u, v
end

"""Return copies of `(u, v)` that satisfy wall boundary conditions on all borders."""
function with_wall_boundaries(u::AbstractMatrix, v::AbstractMatrix)
    uw = copy(float.(u))
    vw = copy(float.(v))
    apply_wall_boundaries!(uw, vw)
    return uw, vw
end

function gradients(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real)
    Nx = size(v, 1)
    Ny = size(u, 2)
    _check_sizes(u, v, Nx, Ny)

    du_dx = similar(float.(u), Nx, Ny)
    dv_dy = similar(float.(v), Nx, Ny)
    du_dy = similar(float.(u), Nx, Ny)
    dv_dx = similar(float.(v), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        du_dx[i, j] = (u[i + 1, j] - u[i, j]) / dx
        dv_dy[i, j] = (v[i, j + 1] - v[i, j]) / dy
        du_dy[i, j] = (u[i + 1, j] - u[i, j]) / dy
        dv_dx[i, j] = (v[i, j + 1] - v[i, j]) / dx
    end

    println("du_dx: ", du_dx)
    println("dv_dy: ", dv_dy)
    println("du_dy: ", du_dy)
    println("dv_dx: ", dv_dx)
    return du_dx, dv_dy, du_dy, dv_dx
end


function diffusive_flux(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real, nu::Real)
    du_dx, dv_dy, du_dy, dv_dx = gradients(u, v, dx, dy)
    flux_diff_u_x = nu * du_dx * dy
    flux_diff_u_y = nu * du_dy * dx
    flux_diff_v_x = nu * dv_dx * dy
    flux_diff_v_y = nu * dv_dy * dx
    return flux_diff_u_x, flux_diff_u_y, flux_diff_v_x, flux_diff_v_y
end


"""
Compute convective fluxes on a 2D staggered (MAC) grid.

Flux definitions:
- flux_conv_x_u = uface_x_u * u * dy
- flux_conv_y_u = vface_on_u * u * dx
- flux_conv_x_v = uface_on_v * v * dy
- flux_conv_y_v = vface_y_v * v * dx
"""

function convective_flux(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real)
    Nx = size(v, 1)
    Ny = size(u, 2)
    _check_sizes(u, v, Nx, Ny)

    T = promote_type(eltype(u), eltype(v), Float64)

    flux_conv_x_u = zeros(T, Nx + 1, Ny)
    flux_conv_y_u = zeros(T, Nx + 1, Ny)
    flux_conv_x_v = zeros(T, Nx, Ny + 1)
    flux_conv_y_v = zeros(T, Nx, Ny + 1)

    # --- u-equation fluxes ---
    for j in 2:Ny-1
        for i in 2:Nx
            # u-face (x-direction) → centered average
            uface_x = 0.5 * (u[i, j] + u[i-1, j])
            flux_conv_x_u[i, j] = uface_x * u[i, j] * dy

            # v interpolated on u-location
            vface = 0.25 * (
                v[i-1, j] + v[i - 1, j + 1] +
                v[i, j]   + v[i, j + 1]
            )
            flux_conv_y_u[i, j] = vface * u[i, j] * dx
        end
    end

    # --- v-equation fluxes ---
    for j in 2:Ny
        for i in 2:Nx-1
            # u interpolated on v-location
            uface = 0.25 * (
                u[i, j-1] + u[i+1, j-1] +
                u[i, j]   + u[i+1, j]
            )
            flux_conv_x_v[i, j] = uface * v[i, j] * dy

            # v-face (y-direction)
            vface_y = 0.5 * (v[i, j] + v[i, j-1])
            flux_conv_y_v[i, j] = vface_y * v[i, j] * dx
        end
    end

    return flux_conv_x_u, flux_conv_y_u, flux_conv_x_v, flux_conv_y_v
end

using Plots

"""
Interpolate staggered (MAC) velocity field to cell centers.
Returns (uc, vc) of size (Nx, Ny).
"""
function interpolate_to_centers(u::AbstractMatrix, v::AbstractMatrix)
    Nx = size(v, 1)
    Ny = size(u, 2)
    _check_sizes(u, v, Nx, Ny)

    uc = zeros(Float64, Nx, Ny)
    vc = zeros(Float64, Nx, Ny)

    for j in 1:Ny
        for i in 1:Nx
            uc[i, j] = 0.5 * (u[i, j] + u[i+1, j])
            vc[i, j] = 0.5 * (v[i, j] + v[i, j+1])
        end
    end

    return uc, vc
end

"""
Plot velocity field stored on MAC grid.
"""
function plot_velocity_field(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real)
    Nx = size(v, 1)
    Ny = size(u, 2)

    uc, vc = interpolate_to_centers(u, v)

    x = collect(dx/2:dx:(Nx-0.5)*dx)
    y = collect(dy/2:dy:(Ny-0.5)*dy)

    X = repeat(x', Ny, 1)'
    Y = repeat(y, 1, Nx)'

    quiver(
        X, Y,
        quiver = (uc, vc),
        xlabel = "x",
        ylabel = "y",
        title = "Velocity Field (MAC Grid)",
        aspect_ratio = :equal,
        legend = false
    )
end

# Example usage:
plot_velocity_field(u, v, dx, dy)
