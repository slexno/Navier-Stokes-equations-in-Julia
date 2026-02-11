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

"""
Compute gradients naturally located at pressure-cell centers:
- `du_dx_center[i,j] = (u[i+1,j] - u[i,j]) / dx`
- `dv_dy_center[i,j] = (v[i,j+1] - v[i,j]) / dy`

Returns `(du_dx_center, dv_dy_center)` with size `(Nx, Ny)`.
"""
function center_normal_gradients(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real)
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

"""Interpolate center values `(Nx, Ny)` to x-normal faces `(Nx+1, Ny)`."""
function interpolate_center_to_xfaces(ϕc::AbstractMatrix)
    Nx, Ny = size(ϕc)
    ϕx = similar(float.(ϕc), Nx + 1, Ny)

    for j in 1:Ny
        ϕx[1, j] = ϕc[1, j]
        for i in 2:Nx
            ϕx[i, j] = 0.5 * (ϕc[i - 1, j] + ϕc[i, j])
        end
        ϕx[Nx + 1, j] = ϕc[Nx, j]
    end

    return ϕx
end

"""Interpolate center values `(Nx, Ny)` to y-normal faces `(Nx, Ny+1)`."""
function interpolate_center_to_yfaces(ϕc::AbstractMatrix)
    Nx, Ny = size(ϕc)
    ϕy = similar(float.(ϕc), Nx, Ny + 1)

    for i in 1:Nx
        ϕy[i, 1] = ϕc[i, 1]
        for j in 2:Ny
            ϕy[i, j] = 0.5 * (ϕc[i, j - 1] + ϕc[i, j])
        end
        ϕy[i, Ny + 1] = ϕc[i, Ny]
    end

    return ϕy
end

"""Cell-centered `u` from face values `(Nx+1,Ny)` -> `(Nx,Ny)`."""
function u_to_centers(u::AbstractMatrix)
    Nx = size(u, 1) - 1
    Ny = size(u, 2)
    uc = similar(float.(u), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        uc[i, j] = 0.5 * (u[i, j] + u[i + 1, j])
    end

    return uc
end

"""Cell-centered `v` from face values `(Nx,Ny+1)` -> `(Nx,Ny)`."""
function v_to_centers(v::AbstractMatrix)
    Nx = size(v, 1)
    Ny = size(v, 2) - 1
    vc = similar(float.(v), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        vc[i, j] = 0.5 * (v[i, j] + v[i, j + 1])
    end

    return vc
end

"""
Compute all gradients required by the request:
- `du_dx_xfaces` and `du_dx_yfaces`
- `dv_dy_xfaces` and `dv_dy_yfaces`
- `du_dy_yfaces`
- `dv_dx_xfaces`
"""
function interpolated_gradients(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real)
    Nx = size(v, 1)
    Ny = size(u, 2)
    _check_sizes(u, v, Nx, Ny)

    du_dx_center, dv_dy_center = center_normal_gradients(u, v, dx, dy)
    du_dx_xfaces = interpolate_center_to_xfaces(du_dx_center)
    du_dx_yfaces = interpolate_center_to_yfaces(du_dx_center)
    dv_dy_xfaces = interpolate_center_to_xfaces(dv_dy_center)
    dv_dy_yfaces = interpolate_center_to_yfaces(dv_dy_center)

    uc = u_to_centers(u)
    vc = v_to_centers(v)

    du_dy_center = similar(uc)
    dv_dx_center = similar(vc)

    for i in 1:Nx
        if Ny == 1
            du_dy_center[i, 1] = 0.0
            continue
        end

        du_dy_center[i, 1] = (uc[i, 2] - uc[i, 1]) / dy
        for j in 2:(Ny - 1)
            du_dy_center[i, j] = (uc[i, j + 1] - uc[i, j - 1]) / (2 * dy)
        end
        du_dy_center[i, Ny] = (uc[i, Ny] - uc[i, Ny - 1]) / dy
    end

    for j in 1:Ny
        if Nx == 1
            dv_dx_center[1, j] = 0.0
            continue
        end

        dv_dx_center[1, j] = (vc[2, j] - vc[1, j]) / dx
        for i in 2:(Nx - 1)
            dv_dx_center[i, j] = (vc[i + 1, j] - vc[i - 1, j]) / (2 * dx)
        end
        dv_dx_center[Nx, j] = (vc[Nx, j] - vc[Nx - 1, j]) / dx
    end

    du_dy_yfaces = interpolate_center_to_yfaces(du_dy_center)
    dv_dx_xfaces = interpolate_center_to_xfaces(dv_dx_center)

    return (
        du_dx_xfaces = du_dx_xfaces,
        du_dx_yfaces = du_dx_yfaces,
        dv_dy_xfaces = dv_dy_xfaces,
        dv_dy_yfaces = dv_dy_yfaces,
        du_dy_yfaces = du_dy_yfaces,
        dv_dx_xfaces = dv_dx_xfaces,
    )
end

"""
Compute diffusive fluxes on faces for both velocity components.

For `u`:
- `Fx_udiff = ν * (du/dx)|xface * ΔSx`, with `ΔSx = dy`
- `Fy_udiff = ν * (du/dy)|yface * ΔSy`, with `ΔSy = dx`

For `v`:
- `Fx_vdiff = ν * (dv/dx)|xface * ΔSx`, with `ΔSx = dy`
- `Fy_vdiff = ν * (dv/dy)|yface * ΔSy`, with `ΔSy = dx`
"""
function diffusive_fluxes(u::AbstractMatrix, v::AbstractMatrix, ν::Real, dx::Real, dy::Real)
    grads = interpolated_gradients(u, v, dx, dy)

    Fx_udiff = ν .* grads.du_dx_xfaces .* dy
    Fy_udiff = ν .* grads.du_dy_yfaces .* dx

    Fx_vdiff = ν .* grads.dv_dx_xfaces .* dy
    Fy_vdiff = ν .* grads.dv_dy_yfaces .* dx

    return (
        Fx_udiff = Fx_udiff,
        Fy_udiff = Fy_udiff,
        Fx_vdiff = Fx_vdiff,
        Fy_vdiff = Fy_vdiff,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    Nx, Ny = 4, 3
    dx, dy = 0.5, 0.25
    ν = 1e-2

    _, u, v = allocate_staggered_fields(Nx, Ny)

    for j in 1:Ny, i in 1:Nx+1
        u[i, j] = i + 0.2j
    end
    for j in 1:Ny+1, i in 1:Nx
        v[i, j] = -0.3i + 0.5j
    end

    fluxes = diffusive_fluxes(u, v, ν, dx, dy)
    @show size(fluxes.Fx_udiff) size(fluxes.Fy_udiff)
    @show size(fluxes.Fx_vdiff) size(fluxes.Fy_vdiff)
end

"""Initialize staggered velocities to zero: `u,v = (0,0)`."""
function initialize_zero_velocity(Nx::Int, Ny::Int)
    u = zeros(Nx + 1, Ny)
    v = zeros(Nx, Ny + 1)
    return u, v
end

"""Average x-face values `(Nx+1,Ny)` to cell centers `(Nx,Ny)`."""
function xfaces_to_centers(ϕx::AbstractMatrix)
    Nx = size(ϕx, 1) - 1
    Ny = size(ϕx, 2)
    ϕc = similar(float.(ϕx), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        ϕc[i, j] = 0.5 * (ϕx[i, j] + ϕx[i + 1, j])
    end

    return ϕc
end

"""Average y-face values `(Nx,Ny+1)` to cell centers `(Nx,Ny)`."""
function yfaces_to_centers(ϕy::AbstractMatrix)
    Nx = size(ϕy, 1)
    Ny = size(ϕy, 2) - 1
    ϕc = similar(float.(ϕy), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        ϕc[i, j] = 0.5 * (ϕy[i, j] + ϕy[i, j + 1])
    end

    return ϕc
end

"""
Compute convective fluxes on faces for both transported components (`u` and `v`).

On x-normal faces: `Fx = Uface * ϕface * dSx` with `dSx = dy`.
On y-normal faces: `Fy = Vface * ϕface * dSy` with `dSy = dx`.

Returned arrays are face-located:
- `Fx_uconv (Nx+1,Ny)`, `Fy_uconv (Nx,Ny+1)`
- `Fx_vconv (Nx+1,Ny)`, `Fy_vconv (Nx,Ny+1)`
"""
function convective_fluxes(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real)
    Nx = size(v, 1)
    Ny = size(u, 2)
    _check_sizes(u, v, Nx, Ny)

    uc = u_to_centers(u)
    vc = v_to_centers(v)

    Uface = interpolate_center_to_xfaces(uc)
    Vface = interpolate_center_to_yfaces(vc)

    u_yface = interpolate_center_to_yfaces(uc)
    v_xface = interpolate_center_to_xfaces(vc)

    Fx_uconv = Uface .* Uface .* dy
    Fy_uconv = Vface .* u_yface .* dx

    Fx_vconv = Uface .* v_xface .* dy
    Fy_vconv = Vface .* Vface .* dx

    return (
        Fx_uconv = Fx_uconv,
        Fy_uconv = Fy_uconv,
        Fx_vconv = Fx_vconv,
        Fy_vconv = Fy_vconv,
    )
end

"""Finite-volume divergence from face fluxes to cell centers `(Nx,Ny)`."""
function flux_divergence(Fx::AbstractMatrix, Fy::AbstractMatrix, dx::Real, dy::Real)
    Nx = size(Fx, 1) - 1
    Ny = size(Fy, 2) - 1
    size(Fx, 2) == Ny || throw(ArgumentError("Fx must have size (Nx+1, Ny)"))
    size(Fy, 1) == Nx || throw(ArgumentError("Fy must have size (Nx, Ny+1)"))

    divF = similar(float.(Fx), Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        divF[i, j] = (Fx[i + 1, j] - Fx[i, j]) / dx + (Fy[i, j + 1] - Fy[i, j]) / dy
    end
    return divF
end

"""Map cell-centered field `(Nx,Ny)` to x-faces `(Nx+1,Ny)` with edge copy."""
function centers_to_xfaces(ϕc::AbstractMatrix)
    return interpolate_center_to_xfaces(ϕc)
end

"""Map cell-centered field `(Nx,Ny)` to y-faces `(Nx,Ny+1)` with edge copy."""
function centers_to_yfaces(ϕc::AbstractMatrix)
    return interpolate_center_to_yfaces(ϕc)
end

"""
Compute intermediate velocity `(u*, v*)` using explicit flux divergence.

Discrete form used at cell centers:
`u* = un - Δt * ∇·(Fconv_u - Fdiff_u)`
`v* = vn - Δt * ∇·(Fconv_v - Fdiff_v)`

Then mapped back to staggered faces.
"""
function intermediate_velocity(
    u::AbstractMatrix,
    v::AbstractMatrix,
    ν::Real,
    dx::Real,
    dy::Real,
    Δt::Real,
)
    conv = convective_fluxes(u, v, dx, dy)
    diff = diffusive_fluxes(u, v, ν, dx, dy)

    div_u = flux_divergence(conv.Fx_uconv .- diff.Fx_udiff, conv.Fy_uconv .- diff.Fy_udiff, dx, dy)
    div_v = flux_divergence(conv.Fx_vconv .- diff.Fx_vdiff, conv.Fy_vconv .- diff.Fy_vdiff, dx, dy)

    uc = u_to_centers(u)
    vc = v_to_centers(v)

    ustar_c = uc .- Δt .* div_u
    vstar_c = vc .- Δt .* div_v

    ustar = centers_to_xfaces(ustar_c)
    vstar = centers_to_yfaces(vstar_c)

    return (
        ustar = ustar,
        vstar = vstar,
        ustar_centers = ustar_c,
        vstar_centers = vstar_c,
        div_u = div_u,
        div_v = div_v,
    )
end
