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
function interpolate_center_to_xfaces(phi_c::AbstractMatrix)
    Nx, Ny = size(phi_c)
    phi_x = similar(float.(phi_c), Nx + 1, Ny)

    for j in 1:Ny
        phi_x[1, j] = phi_c[1, j]
        for i in 2:Nx
            phi_x[i, j] = 0.5 * (phi_c[i - 1, j] + phi_c[i, j])
        end
        phi_x[Nx + 1, j] = phi_c[Nx, j]
    end

    return phi_x
end

"""Interpolate center values `(Nx, Ny)` to y-normal faces `(Nx, Ny+1)`."""
function interpolate_center_to_yfaces(phi_c::AbstractMatrix)
    Nx, Ny = size(phi_c)
    phi_y = similar(float.(phi_c), Nx, Ny + 1)

    for i in 1:Nx
        phi_y[i, 1] = phi_c[i, 1]
        for j in 2:Ny
            phi_y[i, j] = 0.5 * (phi_c[i, j - 1] + phi_c[i, j])
        end
        phi_y[i, Ny + 1] = phi_c[i, Ny]
    end

    return phi_y
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

    # Étape 1 — gradients normaux aux centres de cellules
    #   du/dx = (u_E - u_W)/dx, dv/dy = (v_N - v_S)/dy
    du_dx_center, dv_dy_center = center_normal_gradients(u, v, dx, dy)
    # Étape 2 — interpolation centre -> faces pour placer les dérivées
    # sur les maillages où les flux seront évalués.
    du_dx_xfaces = interpolate_center_to_xfaces(du_dx_center)
    du_dx_yfaces = interpolate_center_to_yfaces(du_dx_center)
    dv_dy_xfaces = interpolate_center_to_xfaces(dv_dy_center)
    dv_dy_yfaces = interpolate_center_to_yfaces(dv_dy_center)

    # Étape 3 — reconstruction des vitesses au centre pour calculer
    # les dérivées croisées du/dy et dv/dx.
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
    # Loi de diffusion (type Fick/Newton) : flux = ν * gradient * surface.
    # Ici dSx = dy sur faces x et dSy = dx sur faces y.
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

"""Discrete Laplacian of face-centered `u` using nearest-neighbor boundary values."""
function _laplacian_u(u::AbstractMatrix, dx::Real, dy::Real)
    Nx1, Ny = size(u)
    lap_u = similar(float.(u))

    for j in 1:Ny, i in 1:Nx1
        u_w = u[max(i - 1, 1), j]
        u_e = u[min(i + 1, Nx1), j]
        u_s = u[i, max(j - 1, 1)]
        u_n = u[i, min(j + 1, Ny)]

        lap_u[i, j] = (u_e - 2u[i, j] + u_w) / dx^2 + (u_n - 2u[i, j] + u_s) / dy^2
    end

    return lap_u
end

"""Discrete Laplacian of face-centered `v` using nearest-neighbor boundary values."""
function _laplacian_v(v::AbstractMatrix, dx::Real, dy::Real)
    Nx, Ny1 = size(v)
    lap_v = similar(float.(v))

    for j in 1:Ny1, i in 1:Nx
        v_w = v[max(i - 1, 1), j]
        v_e = v[min(i + 1, Nx), j]
        v_s = v[i, max(j - 1, 1)]
        v_n = v[i, min(j + 1, Ny1)]

        lap_v[i, j] = (v_e - 2v[i, j] + v_w) / dx^2 + (v_n - 2v[i, j] + v_s) / dy^2
    end

    return lap_v
end

"""Intermediate velocity (predictor): `u* = u + Δt ν ∇²u`, `v* = v + Δt ν ∇²v`."""
function intermediate_velocity(u::AbstractMatrix, v::AbstractMatrix, ν::Real, dt::Real, dx::Real, dy::Real)
    # Étape prédicteur explicite (sans pression) :
    #   u* = u^n + dt * ν * ∇²u^n
    #   v* = v^n + dt * ν * ∇²v^n
    u_star = u .+ dt .* ν .* _laplacian_u(u, dx, dy)
    v_star = v .+ dt .* ν .* _laplacian_v(v, dx, dy)
    return u_star, v_star
end

"""Cell-centered divergence: `∂u/∂x + ∂v/∂y` from staggered velocities."""
function divergence_center(u::AbstractMatrix, v::AbstractMatrix, dx::Real, dy::Real)
    du_dx, dv_dy = center_normal_gradients(u, v, dx, dy)
    return du_dx .+ dv_dy
end

"""
Solve pressure Poisson equation:
`∇²pⁿ⁺¹ = (ρ/Δt) (∂u*/∂x + ∂v*/∂y)`
with homogeneous Neumann boundaries and pressure gauge `p[1,1] = 0`.
"""
function solve_pressure_poisson(rhs::AbstractMatrix, dx::Real, dy::Real;
    maxiter::Int = 2000, tol::Real = 1e-8)
    # On résout ∇²p = rhs par itérations de Gauss-Seidel.
    # Les bords utilisent ici une extension au plus proche voisin
    # (équivalent pratique à un gradient normal nul dans ce schéma simple).
    Nx, Ny = size(rhs)
    p = zeros(float(eltype(rhs)), Nx, Ny)
    idx2 = 1 / dx^2
    idy2 = 1 / dy^2

    for _ in 1:maxiter
        max_change = 0.0

        for j in 1:Ny, i in 1:Nx
            if i == 1 && j == 1
                # Jauge de pression : fixe la constante arbitraire de p.
                continue
            end

            p_w = p[max(i - 1, 1), j]
            p_e = p[min(i + 1, Nx), j]
            p_s = p[i, max(j - 1, 1)]
            p_n = p[i, min(j + 1, Ny)]

            p_new = ((p_w + p_e) * idx2 + (p_s + p_n) * idy2 - rhs[i, j]) / (2idx2 + 2idy2)
            max_change = max(max_change, abs(p_new - p[i, j]))
            p[i, j] = p_new
        end

        max_change < tol && break
    end

    return p
end

"""
Projection step:
`(uⁿ⁺¹ - u*)/Δt = -(1/ρ) ∂pⁿ⁺¹/∂x`,
`(vⁿ⁺¹ - v*)/Δt = -(1/ρ) ∂pⁿ⁺¹/∂y`.
"""
function projection_step(u_star::AbstractMatrix, v_star::AbstractMatrix, p::AbstractMatrix,
    ρ::Real, dt::Real, dx::Real, dy::Real)
    # Étape de projection : on retranche le gradient de pression
    # pour imposer (numériquement) une vitesse finale faiblement divergente.
    Nx, Ny = size(p)
    _check_sizes(u_star, v_star, Nx, Ny)

    u_next = copy(float.(u_star))
    v_next = copy(float.(v_star))

    for j in 1:Ny, i in 2:Nx
        dpdx = (p[i, j] - p[i - 1, j]) / dx
        u_next[i, j] = u_star[i, j] - (dt / ρ) * dpdx
    end

    for j in 2:Ny, i in 1:Nx
        dpdy = (p[i, j] - p[i, j - 1]) / dy
        v_next[i, j] = v_star[i, j] - (dt / ρ) * dpdy
    end

    return u_next, v_next
end

"""Initialize staggered velocities to zero: `u,v = (0,0)`."""
function initialize_zero_velocity(Nx::Int, Ny::Int)
    u = zeros(Nx + 1, Ny)
    v = zeros(Nx, Ny + 1)
    return u, v
end

"""Average x-face values `(Nx+1,Ny)` to cell centers `(Nx,Ny)`."""
function xfaces_to_centers(phi_x::AbstractMatrix)
    Nx = size(phi_x, 1) - 1
    Ny = size(phi_x, 2)
    phi_c = similar(float.(phi_x), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        phi_c[i, j] = 0.5 * (phi_x[i, j] + phi_x[i + 1, j])
    end

    return phi_c
end

"""Average y-face values `(Nx,Ny+1)` to cell centers `(Nx,Ny)`."""
function yfaces_to_centers(phi_y::AbstractMatrix)
    Nx = size(phi_y, 1)
    Ny = size(phi_y, 2) - 1
    phi_c = similar(float.(phi_y), Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        phi_c[i, j] = 0.5 * (phi_y[i, j] + phi_y[i, j + 1])
    end

    return phi_c
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
    # Flux convectifs quadratiques : F = (vitesse advectante) * (quantité transportée) * surface.
    # Les champs sont interpolés vers les faces pour respecter la géométrie MAC.
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
    # Divergence volumes finis :
    #   div(F) = (F_e - F_w)/dx + (F_n - F_s)/dy
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
function intermediate_velocity_flux_form(
    u::AbstractMatrix,
    v::AbstractMatrix,
    ν::Real,
    dx::Real,
    dy::Real,
    Δt::Real,
)
    # Formulation conservation explicite :
    #   u* = u^n - dt * div(F_conv - F_diff)
    #   v* = v^n - dt * div(F_conv - F_diff)
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

function _print_matrix(name::AbstractString, A::AbstractMatrix)
    println("\n" * name * " (size=" * string(size(A)) * ")")
    display(A)
end


    
function demo_staggered_grid()
    # ---------------------------
    # Cas de démonstration :
    # 1) construction d'un état initial
    # 2) calcul des opérateurs/flux
    # 3) étape prédicteur
    # 4) Poisson pression
    # 5) projection finale
    # ---------------------------
    Nx, Ny = 3, 3
    dx, dy = 0.5, 0.5
    ν = 1e-2
    ρ = 1.0
    dt = 0.05

    p, u, v = allocate_staggered_fields(Nx, Ny)
    for j in 1:Ny, i in 1:Nx+1
        u[i, j] = i + 0.2j
    end
    for j in 1:Ny+1, i in 1:Nx
        v[i, j] = -0.3i + 0.5j
    end

    _print_matrix("allocate_staggered_fields -> p", p)
    _print_matrix("allocate_staggered_fields -> u", u)
    _print_matrix("allocate_staggered_fields -> v", v)

    du_dx_center, dv_dy_center = center_normal_gradients(u, v, dx, dy)
    _print_matrix("center_normal_gradients -> du_dx_center", du_dx_center)
    _print_matrix("center_normal_gradients -> dv_dy_center", dv_dy_center)

    _print_matrix("interpolate_center_to_xfaces(du_dx_center)", interpolate_center_to_xfaces(du_dx_center))
    _print_matrix("interpolate_center_to_yfaces(du_dx_center)", interpolate_center_to_yfaces(du_dx_center))
    _print_matrix("u_to_centers(u)", u_to_centers(u))
    _print_matrix("v_to_centers(v)", v_to_centers(v))

    grads = interpolated_gradients(u, v, dx, dy)
    _print_matrix("interpolated_gradients -> du_dx_xfaces", grads.du_dx_xfaces)
    _print_matrix("interpolated_gradients -> du_dx_yfaces", grads.du_dx_yfaces)
    _print_matrix("interpolated_gradients -> dv_dy_xfaces", grads.dv_dy_xfaces)
    _print_matrix("interpolated_gradients -> dv_dy_yfaces", grads.dv_dy_yfaces)
    _print_matrix("interpolated_gradients -> du_dy_yfaces", grads.du_dy_yfaces)
    _print_matrix("interpolated_gradients -> dv_dx_xfaces", grads.dv_dx_xfaces)

    diff = diffusive_fluxes(u, v, ν, dx, dy)
    _print_matrix("diffusive_fluxes -> Fx_udiff", diff.Fx_udiff)
    _print_matrix("diffusive_fluxes -> Fy_udiff", diff.Fy_udiff)
    _print_matrix("diffusive_fluxes -> Fx_vdiff", diff.Fx_vdiff)
    _print_matrix("diffusive_fluxes -> Fy_vdiff", diff.Fy_vdiff)

    _print_matrix("_laplacian_u(u)", _laplacian_u(u, dx, dy))
    _print_matrix("_laplacian_v(v)", _laplacian_v(v, dx, dy))

    u_star, v_star = intermediate_velocity(u, v, ν, dt, dx, dy)
    _print_matrix("intermediate_velocity -> u_star", u_star)
    _print_matrix("intermediate_velocity -> v_star", v_star)

    div_center = divergence_center(u_star, v_star, dx, dy)
    _print_matrix("divergence_center(u_star, v_star)", div_center)

    rhs = (ρ / dt) .* div_center
    _print_matrix("Poisson RHS", rhs)

    p_next = solve_pressure_poisson(rhs, dx, dy)
    _print_matrix("solve_pressure_poisson -> p_next", p_next)

    u_next, v_next = projection_step(u_star, v_star, p_next, ρ, dt, dx, dy)
    _print_matrix("projection_step -> u_next", u_next)
    _print_matrix("projection_step -> v_next", v_next)

    u0, v0 = initialize_zero_velocity(Nx, Ny)
    _print_matrix("initialize_zero_velocity -> u0", u0)
    _print_matrix("initialize_zero_velocity -> v0", v0)

    uc = u_to_centers(u)
    vc = v_to_centers(v)
    _print_matrix("xfaces_to_centers(u)", xfaces_to_centers(u))
    _print_matrix("yfaces_to_centers(v)", yfaces_to_centers(v))
    _print_matrix("centers_to_xfaces(uc)", centers_to_xfaces(uc))
    _print_matrix("centers_to_yfaces(vc)", centers_to_yfaces(vc))

    conv = convective_fluxes(u, v, dx, dy)
    _print_matrix("convective_fluxes -> Fx_uconv", conv.Fx_uconv)
    _print_matrix("convective_fluxes -> Fy_uconv", conv.Fy_uconv)
    _print_matrix("convective_fluxes -> Fx_vconv", conv.Fx_vconv)
    _print_matrix("convective_fluxes -> Fy_vconv", conv.Fy_vconv)

    _print_matrix("flux_divergence(Fx_uconv, Fy_uconv)", flux_divergence(conv.Fx_uconv, conv.Fy_uconv, dx, dy))
    _print_matrix("flux_divergence(Fx_vconv, Fy_vconv)", flux_divergence(conv.Fx_vconv, conv.Fy_vconv, dx, dy))

    iv_flux = intermediate_velocity_flux_form(u, v, ν, dx, dy, dt)
    _print_matrix("intermediate_velocity_flux_form -> ustar", iv_flux.ustar)
    _print_matrix("intermediate_velocity_flux_form -> vstar", iv_flux.vstar)
    _print_matrix("intermediate_velocity_flux_form -> ustar_centers", iv_flux.ustar_centers)
    _print_matrix("intermediate_velocity_flux_form -> vstar_centers", iv_flux.vstar_centers)
    _print_matrix("intermediate_velocity_flux_form -> div_u", iv_flux.div_u)
    _print_matrix("intermediate_velocity_flux_form -> div_v", iv_flux.div_v)
end

demo_staggered_grid()