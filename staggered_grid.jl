"""
Utilities for a 2D staggered (MAC) grid discretization.

Grid arrangement:
- pressure `p`: `(Nx, Ny)` cell centers
- x-velocity `u`: `(Nx + 1, Ny)` x-normal faces
- y-velocity `v`: `(Nx, Ny + 1)` y-normal faces
"""

"""Create zero-initialized staggered fields `(p, u, v)`."""
function allocate_staggered_fields(Nx::Int, Ny::Int)
    p = zeros(Nx, Ny)
    u = zeros(Nx + 1, Ny)
    v = zeros(Nx, Ny + 1)
    return p, u, v
end

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

    du_dx_center = similar(float.(u), Nx, Ny)
    dv_dy_center = similar(float.(v), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        du_dx_center[i, j] = (u[i + 1, j] - u[i, j]) / dx
        dv_dy_center[i, j] = (v[i, j + 1] - v[i, j]) / dy
    end

    return du_dx_center, dv_dy_center
end

du_dx_center, dv_dy_center = gradients(u, v, dx, dy)


