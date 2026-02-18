"""
Utilities for a 2D staggered (MAC) grid discretization.

Grid arrangement:
- pressure `p`: `(Nx, Ny)` cell centers
- x-velocity `u`: `(Nx + 1, Ny)` x-normal faces
- y-velocity `v`: `(Nx, Ny + 1)` y-normal faces
"""

using Printf
using Plots

Nx, Ny = 5,5
dx, dy = 0.01, 0.1
dt =0.01

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
    u[end, :] .= 1.0
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

    u, v = with_wall_boundaries(u, v)
    println(u,v)
    du_dx = similar(float.(u), Nx, Ny)
    dv_dy = similar(float.(v), Nx, Ny)
    du_dy = similar(float.(u), Nx, Ny)
    dv_dx = similar(float.(v), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        du_dx[i, j] = (u[i + 1, j] - u[i, j]) / dx
        dv_dy[i, j] = (v[i, j + 1] - v[i, j]) / dy
        print(v[i, j + 1] , v[i, j])
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

flux_diff_u_x, flux_diff_u_y, flux_diff_v_x, flux_diff_v_y = diffusive_flux(u, v, dx, dy, nu)

# Plot contourf
p = contourf(flux_diff_u_x', xlabel="x", ylabel="y",
             title="Diffusive Flux (u-x direction)", color=:viridis)

savefig(p, "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\diffusive_flux_contour.png")
@info "Figure saved → diffusive_flux_contour.png"

contourf(flux_diff_u_y', xlabel="x", ylabel="y",
         title="Diffusive Flux (u-y direction)", color=:viridis)



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


flux_conv_x_u, flux_conv_y_u, flux_conv_x_v, flux_conv_y_v = convective_flux(u, v, dx, dy)

# Plot contourf
p = contourf(flux_conv_x_u', xlabel="x", ylabel="y",
             title="Convective Flux (u-x direction)", color=:viridis)

savefig(p, "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\convective_flux_u_x_contour.png")
@info "Figure saved → convective_flux_u_x_contour.png"

contourf(flux_conv_y_u', xlabel="x", ylabel="y",
         title="Convective Flux (u-y direction)", color=:viridis)

savefig(p, "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\convective_flux_u_y_contour.png")
@info "Figure saved → convective_flux_u_y_contour.png"

contourf(flux_conv_x_v', xlabel="x", ylabel="y",
         title="Convective Flux (v-x direction)", color=:viridis)

savefig(p, "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\convective_flux_v_x_contour.png")
@info "Figure saved → convective_flux_v_x_contour.png"

contourf(flux_conv_y_v', xlabel="x", ylabel="y",
         title="Convective Flux (v-y direction)", color=:viridis)

savefig(p, "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\convective_flux_v_y_contour.png")
@info "Figure saved → convective_flux_v_y_contour.png"




function intermediate_velocity(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real, nu::Real)
    flux_diff_u_x, flux_diff_u_y, flux_diff_v_x, flux_diff_v_y = diffusive_flux(u, v, dx, dy, nu)
    flux_conv_x_u, flux_conv_y_u, flux_conv_x_v, flux_conv_y_v = convective_flux(u, v, dx, dy)

    # Compute intermediate velocities (without pressure gradient)
    u_star = similar(float.(u))
    v_star = similar(float.(v))

    for j in 2:size(u, 2)-1
        for i in 2:size(u, 1)-1
            u_star[i, j] = u[i, j] + (flux_diff_u_x[i, j] + flux_diff_u_y[i, j]) - (flux_conv_x_u[i, j] + flux_conv_y_u[i, j])
        end
    end

    for j in 2:size(v, 2)-1
        for i in 2:size(v, 1)-1
            v_star[i, j] = v[i, j] + (flux_diff_v_x[i, j] + flux_diff_v_y[i, j]) - (flux_conv_x_v[i, j] + flux_conv_y_v[i, j])
        end
    end

    return u_star, v_star
end

function solve_poisson_equation_for_pressure(p::AbstractMatrix, rho::AbstractMatrix, dx::Real, dy::Real, u_star, v_star, dt::Real)
    Nx = size(p, 1)
    Ny = size(p, 2)
    
    # Right-hand side: rho/dt * (du_star/dx + dv_star/dy)
    rhs = zeros(Float64, Nx, Ny)
    
    for j in 1:Ny
        for i in 1:Nx
            du_star_dx = (u_star[i + 1, j] - u_star[i, j]) / dx
            dv_star_dy = (v_star[i, j + 1] - v_star[i, j]) / dy
            rhs[i, j] = rho[i, j] / dt * (du_star_dx + dv_star_dy)
        end
    end
    
    # Solve Laplacian equation: ∇²p = rhs using iterative method (Jacobi iteration)
    p_new = copy(float.(p))
    
    for iter in 1:100  # Fixed number of iterations
        for j in 2:Ny-1
            for i in 2:Nx-1
                p_new[i, j] = 0.25 * (
                    p_new[i+1, j] + p_new[i-1, j] +
                    p_new[i, j+1] + p_new[i, j-1] -
                    dx * dy * rhs[i, j]
                )
            end
        end
    end
    
    return p_new
end


function projection_step(u_star::AbstractMatrix, v_star::AbstractMatrix, p::AbstractMatrix, rho::AbstractMatrix, dx::Real, dy::Real, dt::Real)
    Nx = size(p, 1)
    Ny = size(p, 2)
    
    u_new = copy(float.(u_star))
    v_new = copy(float.(v_star))
    
    # u-component: u^{n+1} = u_star - (dt/rho) * ∂p/∂x
    for j in 1:Ny
        for i in 2:Nx
            dp_dx = (p[i, j] - p[i-1, j]) / dx
            u_new[i, j] = u_star[i, j] - (dt / rho[i, j]) * dp_dx
        end
    end
    
    # v-component: v^{n+1} = v_star - (dt/rho) * ∂p/∂y
    for j in 2:Ny
        for i in 1:Nx
            dp_dy = (p[i, j] - p[i, j-1]) / dy
            v_new[i, j] = v_star[i, j] - (dt / rho[i, j]) * dp_dy
        end
    end
    
    return u_new, v_new
end


