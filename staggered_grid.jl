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
dt =0.0001

p = zeros(Nx, Ny)
u = zeros(Nx + 1, Ny)
v = zeros(Nx, Ny + 1)

nu = 1e-3



function _check_sizes(u, v, Nx::Int, Ny::Int)
    size(u) == (Nx + 1, Ny) || throw(ArgumentError("u must have size ($(Nx+1), $Ny)"))
    size(v) == (Nx, Ny + 1) || throw(ArgumentError("v must have size ($Nx, $(Ny+1))"))
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
        # x-derivatives: only the first and last cell-centered differences are doubled
        factor_x = (i == 1 || i == Nx) ? 2.0 : 1.0
        # y-derivatives: only the first and last cell-centered differences are doubled
        factor_y = (j == 1 || j == Ny) ? 2.0 : 1.0

        du_dx[i, j] = factor_x * (u[i + 1, j] - u[i, j]) / dx
        dv_dy[i, j] = factor_y * (v[i, j + 1] - v[i, j]) / dy

        if j < Ny
            du_dy[i, j] = factor_y * (u[i, j + 1] - u[i, j]) / dy
        else
            du_dy[i, j] = factor_y * (u[i, j] - u[i, j - 1]) / dy
        end

        if i < Nx
            dv_dx[i, j] = factor_x * (v[i + 1, j] - v[i, j]) / dx
        else
            dv_dx[i, j] = factor_x * (v[i, j] - v[i - 1, j]) / dx
        end
    end

    return du_dx, dv_dy, du_dy, dv_dx
end



function diffusive_flux(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real, nu::Real)

    du_dx, dv_dy, du_dy, dv_dx = gradients(u, v, dx, dy)

    Nx, Ny = size(du_dx)

    flux_diff_u_x = zeros(Nx, Ny)
    flux_diff_u_y = zeros(Nx, Ny)
    flux_diff_v_x = zeros(Nx, Ny)
    flux_diff_v_y = zeros(Nx, Ny)

    # Divergence of diffusive flux (i.e. Laplacian reconstruction)
    for j in 1:Ny, i in 1:Nx
        dudx_x = i == 1  ? (du_dx[i + 1, j] - du_dx[i, j]) / dx :
                 i == Nx ? (du_dx[i, j] - du_dx[i - 1, j]) / dx :
                           (du_dx[i, j] - du_dx[i - 1, j]) / dx

        dudy_y = j == 1  ? (du_dy[i, j + 1] - du_dy[i, j]) / dy :
                 j == Ny ? (du_dy[i, j] - du_dy[i, j - 1]) / dy :
                           (du_dy[i, j] - du_dy[i, j - 1]) / dy

        dvdx_x = i == 1  ? (dv_dx[i + 1, j] - dv_dx[i, j]) / dx :
                 i == Nx ? (dv_dx[i, j] - dv_dx[i - 1, j]) / dx :
                           (dv_dx[i, j] - dv_dx[i - 1, j]) / dx

        dvdy_y = j == 1  ? (dv_dy[i, j + 1] - dv_dy[i, j]) / dy :
                 j == Ny ? (dv_dy[i, j] - dv_dy[i, j - 1]) / dy :
                           (dv_dy[i, j] - dv_dy[i, j - 1]) / dy

        flux_diff_u_x[i,j] = nu * dudx_x
        flux_diff_u_y[i,j] = nu * dudy_y
        flux_diff_v_x[i,j] = nu * dvdx_x
        flux_diff_v_y[i,j] = nu * dvdy_y
    end

    return flux_diff_u_x, flux_diff_u_y, flux_diff_v_x, flux_diff_v_y
end


"""
Advance one explicit diffusion step on MAC-grid velocities.

- u is diffused on `(Nx+1, Ny)` x-faces
- v is diffused on `(Nx, Ny+1)` y-faces

Boundary choices used here:
- left u boundary is kept fixed (Dirichlet inlet)
- other boundaries are copied from adjacent interior values (homogeneous Neumann)
"""
function diffuse_velocity_step!(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real, dt::Real, nu::Real)
    Nx = size(v, 1)
    Ny = size(u, 2)
    _check_sizes(u, v, Nx, Ny)

    un = copy(u)
    vn = copy(v)

    inv_dx2 = 1.0 / dx^2
    inv_dy2 = 1.0 / dy^2

    # Diffusion on u faces (interior in x: 2..Nx, in y: 2..Ny-1)
    for j in 2:Ny-1, i in 2:Nx
        lap_u = (un[i + 1, j] - 2.0 * un[i, j] + un[i - 1, j]) * inv_dx2 +
                (un[i, j + 1] - 2.0 * un[i, j] + un[i, j - 1]) * inv_dy2
        u[i, j] = un[i, j] + dt * nu * lap_u
    end

    # Diffusion on v faces (interior in x: 2..Nx-1, in y: 2..Ny)
    for j in 2:Ny, i in 2:Nx-1
        lap_v = (vn[i + 1, j] - 2.0 * vn[i, j] + vn[i - 1, j]) * inv_dx2 +
                (vn[i, j + 1] - 2.0 * vn[i, j] + vn[i, j - 1]) * inv_dy2
        v[i, j] = vn[i, j] + dt * nu * lap_v
    end

    # Neumann-like copies on non-inlet boundaries
    u[end, :] .= u[end - 1, :]
    u[:, 1] .= u[:, 2]
    u[:, end] .= u[:, end - 1]

    v[1, :] .= v[2, :]
    v[end, :] .= v[end - 1, :]
    v[:, 1] .= v[:, 2]
    v[:, end] .= v[:, end - 1]

    return nothing
end



flux_diff_u_x, flux_diff_u_y, flux_diff_v_x, flux_diff_v_y = diffusive_flux(u, v, dx, dy, nu)

# Plot contourf
p = contourf(flux_diff_u_x', xlabel="x", ylabel="y",
             title="Diffusive Flux (u-x direction)", color=:viridis)

savefig(p, "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\diffusive_flux_contour.png")
@info "Figure saved → diffusive_flux_contour.png"

contourf(flux_diff_u_y', xlabel="x", ylabel="y",
         title="Diffusive Flux (u-y direction)", color=:viridis)




# ================================
# Time evolution of diffusive flux
# ================================

Nt = 200  # number of time steps

# Inlet (départ) at left wall only
u[1, :] .= 1.0

# Plot at first iteration (t = dt)
first_flux_diff_u_x, first_flux_diff_u_y, first_flux_diff_v_x, first_flux_diff_v_y =
    diffusive_flux(u, v, dx, dy, nu)

p_first = contourf(first_flux_diff_u_x', xlabel="x", ylabel="y",
                   title="Diffusive Flux u-x (iteration 1)", color=:viridis)
savefig(p_first, "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\diffusive_flux_iteration1.png")
@info "Figure saved → diffusive_flux_iteration1.png"


# Fix color scale across all frames (avoids visual masking when values evolve)
umin = minimum(first_flux_diff_u_x)
umax = maximum(first_flux_diff_u_x)

u_tmp = copy(u)
v_tmp = copy(v)
for _ in 1:Nt
    u_tmp[1, :] .= 1.0
    diffuse_velocity_step!(u_tmp, v_tmp, dx, dy, dt, nu)
    fx_tmp, _, _, _ = diffusive_flux(u_tmp, v_tmp, dx, dy, nu)
    umin = min(umin, minimum(fx_tmp))
    umax = max(umax, maximum(fx_tmp))
end





anim = @animate for n in 1:Nt
    global u, v   # <-- FIX

    # Keep only the left boundary condition fixed to 1
    u[1, :] .= 1.0

    flux_diff_u_x, flux_diff_u_y, flux_diff_v_x, flux_diff_v_y =
        diffusive_flux(u, v, dx, dy, nu)

    diffuse_velocity_step!(u, v, dx, dy, dt, nu)

    p1 = contourf(flux_diff_u_x', xlabel="x", ylabel="y",
                  title="Diffusive Flux u-x (t = $(round(n*dt, digits=4)))",
                  color=:viridis, clims=(umin, umax))
    p2 = contourf(flux_diff_u_y', xlabel="x", ylabel="y", title="Diffusive Flux u-y", color=:viridis)
    p3 = contourf(flux_diff_v_x', xlabel="x", ylabel="y", title="Diffusive Flux v-x", color=:viridis)
    p4 = contourf(flux_diff_v_y', xlabel="x", ylabel="y", title="Diffusive Flux v-y", color=:viridis)

    plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700))
end


gif(anim,
    "C:\\Users\\bello\\Documents\\ecole\\Aero_4\\semestre_2\\Julia\\diffusive_flux_time.gif",
    fps=30)

@info "Animation saved → diffusive_flux_time.gif"





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
