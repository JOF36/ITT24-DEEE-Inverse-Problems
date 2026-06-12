#%%
import numpy as np
import matplotlib.pyplot as plt
import os

#%%

def save_matrix_image(A, iteration, max_iters, folder="snapshots"):

    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.imshow(A)
    plt.colorbar()

    pct = int(100 * iteration / max_iters)

    plt.title(f"Learned A ({pct}% complete)")
    plt.tight_layout()

    filename = os.path.join(
        folder,
        f"A_iter_{iteration:05d}_{pct:03d}pct.png"
    )

    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved {filename}")

# ============================================================
# True conductance matrix
# ============================================================

def generate_matrix():

    c = 1.0 / 0.397

    c01 = c / 0.01
    c02 = 0.0
    c03 = c / 0.015
    c12 = c / 0.015
    c13 = 0.0
    c23 = 0.0

    A_true = np.array([
        [0.0, c01, c02, c03],
        [c01, 0.0, c12, c13],
        [c02, c12, 0.0, c23],
        [c03, c13, c23, 0.0]
    ])

    return A_true


# ============================================================
# Voltage samples
# ============================================================

def generate_reference(seed=None):

    rng = np.random.default_rng(seed)

    return np.array([
        240.0,
        239.8 + 0.10 * rng.standard_normal(),
        239.6 + 0.10 * rng.standard_normal(),
        239.9 + 0.10 * rng.standard_normal(),
    ])


# ============================================================
# Forward operator
#
# I_i = sum_j A_ij (v_i - v_j)
# ============================================================

# def Phi(A, v):

#     n = len(v)

#     I = np.zeros(n)

#     for i in range(n):
#         for j in range(n):
#             I[i] += A[i, j] * (v[i] - v[j])

#     return I


# def Phi2(A, v):

#     n = len(v)

#     I = np.zeros(n)

#     for i in range(1, n):
#         for j in range(n):
#             I[i] += A[i, j] * (v[j] - v[i])

#     I[0] = np.sum(I[1:])

#     return I

# ============================================================
# Matrix form of Phi
# ============================================================

def Phi(A, v):

    ones = np.ones(len(v))

    B = np.outer(ones, v) - np.outer(v, ones)

    # d = -ones.copy()
    # d[0] = 1
    # C = np.diag(d)

    # I = np.diag(C @ A @ B)
    I = np.diag(A @ B)

    return I


# ============================================================
# Adjoint
# ============================================================

def PhiAdjoint(y, v):

    ones = np.ones(len(v))

    M = np.outer(v, ones) - np.outer(ones, v)

    return (np.diag(y) @ M).T


# ============================================================
# Dataset
# ============================================================

def generate_data(N=100):

    A_true = generate_matrix()

    V = []
    I = []

    for k in range(N):

        v = generate_reference(k)

        V.append(v)
        I.append(Phi(A_true, v))

    return V, I


# ============================================================
# Objective
# ============================================================

def F(A, V, I):

    loss = 0.0

    for v, i in zip(V, I):

        r = Phi(A, v) - i

        loss += 0.5*np.sum(r**2)

    return loss


# ============================================================
# Gradient
# ============================================================

def G(A, V, I):

    grad = np.zeros_like(A)

    for v, i in zip(V, I):

        r = Phi(A, v) - i

        grad += PhiAdjoint(r, v)

    return grad


# ============================================================
# Projection onto symmetric zero-diagonal matrices
# ============================================================

def project(A):

    A = 0.5 * (A + A.T)

    np.fill_diagonal(A, 0.0)

    return A


# ============================================================
# Finite difference gradient check
# ============================================================

def gradient_check(A, V, I):

    eps = 1e-6

    D = np.random.randn(*A.shape)
    D = project(D)

    lhs = (
        F(A + eps * D, V, I)
        - F(A, V, I)
    ) / eps

    rhs = np.sum(G(A, V, I) * D)

    print("Finite difference check")
    print("lhs =", lhs)
    print("rhs =", rhs)
    print("difference =", abs(lhs - rhs))


# ============================================================
# Gradient descent
# ============================================================

def gradient_descent(
    A,
    V,
    I,
    lr,
    max_iters=10
):

    save_points = {
        int(max_iters * 0.2),
        int(max_iters * 0.4),
        int(max_iters * 0.6),
        int(max_iters * 0.8),
        int(max_iters * 1.0),
    }

    for it in range(max_iters):

        loss = F(A, V, I)

        if it % 10 == 0:
            print(f"Iter {it:5d}   Loss = {loss:.6e}")

        grad = G(A, V, I)

        A -= lr * grad

        A = project(A)

        if (it + 1) in save_points:
            save_matrix_image(
                A,
                iteration=it + 1,
                max_iters=max_iters
            )

    return A


#%%

# ============================================================
# Main
# ============================================================

V, I = generate_data(N=100)

# Ihat = I[0]
# Itilde = Phi2(generate_matrix(), V[0])
# Ibar = Phi_matrix(generate_matrix(), V[0])

# print('Ihat', Ihat)
# print('Itilde', Itilde)
# print('Ibar', Ibar)

A_true = generate_matrix()

rng = np.random.default_rng(0)

A_init = np.random.uniform(0, 400, size=A_true.shape)

A_init = project(A_init)
plt.imshow(A_init)
plt.colorbar()
plt.title("Initial A")
plt.show()

plt.imshow(A_true)
plt.colorbar()
plt.title("True A")
plt.show()


# A_init = A_true + 5000.0 * rng.standard_normal(A_true.shape)
# A_init = project(A_init)

gradient_check(A_init, V, I)


#%%
A_learned = gradient_descent(
    A_init,
    V,
    I,
    lr=1e-2,
    max_iters=1000
)

#%%

np.set_printoptions(precision=2)

print("\nTrue A")
print(A_true)

print("\nLearned A")
print(A_learned)

print("\nError")
print(A_true - A_learned)
print(np.linalg.norm(A_true - A_learned)/np.linalg.norm(A_true))

#%%

plt.figure()
plt.imshow(A_true)
plt.colorbar()
plt.title("True A")

plt.figure()
plt.imshow(A_learned)
plt.colorbar()
plt.title("Learned A")


plt.figure()
plt.imshow(A_init)
plt.colorbar()
plt.title("Initial A")


plt.figure()
plt.imshow(A_true - A_learned)
plt.colorbar()
plt.title("Error")

plt.show()
# %%
