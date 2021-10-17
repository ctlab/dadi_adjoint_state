import torch
import Integration_shared, Integration


# def tridiag(a, b, c, r, u, n):
#     bet = b[0].data
#     gam = torch.zeros(n)
#     u[0] = r[0] / bet
#     for j in range(1, n):  # (j=1; j <= n-1; j++){
#         gam[j] = c[j-1] / bet
#         bet.data = b[j] - a[j] * gam[j]
#         u[j] = (r[j]-a[j] * u[j-1].data) / bet
#
#     for j in range(n-2, -1, -1):  # (j=(n-2); j >= 0; j--){
#         u[j].data -= gam[j+1] * u[j+1]
#     return u


def implicit_1Dx(phi, xx, nu, gamma, h, beta, dt, use_delj_trick):
    dx = torch.diff(xx)
    L = xx.shape[0]
    dfactor = torch.zeros(L)
    xInt = torch.zeros(L - 1)
    V = torch.zeros(L)
    MInt = torch.zeros(L - 1)
    VInt = torch.zeros(L - 1)
    delj = torch.zeros(L - 1)
    a = torch.zeros(L)
    c = torch.zeros(L)
    b = torch.full(L, 1./dt)
    Integration_shared.compute_dfactor(dx, L, dfactor)
    Integration_shared.compute_xInt(xx, L, xInt)
    Mfirst = Integration_shared.Mfunc1D(xx[0], gamma, h)
    Mlast = Integration_shared.Mfunc1D(xx[L - 1], gamma, h)
    V = Integration_shared.Vfunc_beta(xx, nu, beta)
    MInt = Integration_shared.Mfunc1D(xInt, gamma, h)
    VInt = Integration_shared.Vfunc_beta(xInt, nu, beta)
    Integration_shared.compute_delj(dx, MInt, VInt, L, delj, use_delj_trick)
    Integration_shared.compute_abc_nobc(dx, dfactor, delj, MInt, V, L, a, b, c)

    r = phi / dt

    # Boundary conditions
    if Mfirst <= 0:
        b[0] += (0.5 / nu - Mfirst) * 2. / dx[0]
    if (Mlast >= 0):
        b[L - 1] += -(-0.5 / nu - Mlast) * 2. / dx[L - 2]

    Integration.tridiag(a, b, c, r, phi, L)

