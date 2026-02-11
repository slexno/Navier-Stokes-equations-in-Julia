# Gauss-Seidel Method Implementation in Julia
x0 =  zeros(3) 

itermax = 1000
B = [15.0; 10.0; 10.0]

function gauss_seidel(A, B, x0, iter)
    x = copy(x0)
    for k in 1:iter
        for i in 1:length(B)
            for j in 1:length(B)
                if j != i
                    B[i] -= A[i, j] * x[j]
                end
            end
            x[i] = B[i] / A[i, i]
        end
    end
    return x
end




nx = 3
ny = 3
h=1

using SparseArrays


dirichlet = Dict{Tuple{Int,Int},Float64}()

for i in 1:3
    dirichlet[(i,1)] = 0.0
    dirichlet[(i,3)] = 0.0
end

for j in 1:3
    dirichlet[(1,j)] = 0.0
    dirichlet[(3,j)] = 0.0
end



function laplacian(Nx, Ny, dx, dy; dirichlet)
    N = Nx * Ny
    dx2, dy2 = dx^2, dy^2

    idx(i,j) = (j-1)*Nx + i
    I = Int[]; J = Int[]; V = Float64[]

    for i in 1:Nx, j in 1:Ny
        k = idx(i,j)

        if haskey(dirichlet, (i,j))
            push!(I,k); push!(J,k); push!(V,1.0)
            continue
        end

        c = 0.0

        for (ni,nj,coef) in (
            (i+1,j,1/dx2), (i-1,j,1/dx2),
            (i,j+1,1/dy2), (i,j-1,1/dy2)
        )
            if 1 ≤ ni ≤ Nx && 1 ≤ nj ≤ Ny
                push!(I,k); push!(J,idx(ni,nj)); push!(V,coef)
                c -= coef
            end
        end

        push!(I,k); push!(J,k); push!(V,c)
    end

    sparse(I,J,V,N,N)
end
A = laplacian(nx, ny, h, h,dirichlet=dirichlet)


using SparseArrays

# Jauge de pression (NON physique, juste numérique)
gauge = Dict{Tuple{Int,Int},Float64}()


for i in 1:3
    gauge[(i,1)] = 0.0
    gauge[(i,3)] = 0.0
end



function laplacian(Nx, Ny, dx, dy; gauge)
    N = Nx * Ny
    dx2, dy2 = dx^2, dy^2

    idx(i,j) = (j-1)*Nx + i
    I = Int[]; J = Int[]; V = Float64[]

    for i in 1:Nx, j in 1:Ny
        k = idx(i,j)

        # Jauge de pression (1 seule équation fixée)
        if haskey(gauge, (i,j))
            push!(I,k); push!(J,k); push!(V,1.0)
            continue
        end

        c = 0.0

        # Stencil 5 points, Neumann homogène implicite
        for (ni,nj,coef) in (
            (i+1,j,1/dx2), (i-1,j,1/dx2),
            (i,j+1,1/dy2), (i,j-1,1/dy2)
        )
            if 1 ≤ ni ≤ Nx && 1 ≤ nj ≤ Ny
                push!(I,k); push!(J,idx(ni,nj)); push!(V,coef)
                c -= coef
            end
        end

        push!(I,k); push!(J,k); push!(V,c)
    end

    sparse(I,J,V,N,N)
end

# Exemple 3×3
nx, ny = 3, 3
h = 2

A = laplacian(nx, ny, h, h; gauge=gauge)