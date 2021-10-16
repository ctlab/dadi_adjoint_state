import torch
from dadi_torch import Integration, Integration_shared


def implicit_2Dx(phi, xx, yy, nu1, m12, gamma1, h1, dt, use_delj_trick):
    L = phi.shape[0]
    M = phi.shape[1]
    dx = torch.diff(xx)
    dfactor = torch.zeros(L)
    xInt = torch.zeros(L - 1)

    Integration_shared.compute_dfactor(dx, L, dfactor)
    Integration_shared.compute_xInt(xx, L, xInt)

    MInt = torch.zeros(L - 1)
    V = torch.zeros(L)
    VInt = torch.zeros(L - 1)
    delj = torch.zeros(L - 1)

    for ii in range(0, L):
        V[ii] = Integration_shared.Vfunc(xx[ii], nu1)
    for ii in range(0, L-1):
        VInt[ii] = Integration_shared.Vfunc(xInt[ii], nu1)

    a = torch.zeros(L)
    b = torch.zeros(L)
    c = torch.zeros(L)
    r = torch.zeros(L)
    temp = torch.zeros(L)

    for jj in range(0, M):
        y = yy[jj]
        Mfirst = Integration_shared.Mfunc2D(xx[0], y, m12, gamma1, h1)
        Mlast = Integration_shared.Mfunc2D(xx[L - 1], y, m12, gamma1, h1)
        for ii in range(0, L-1):
            MInt[ii] = Integration_shared.Mfunc2D(xInt[ii], y, m12, gamma1, h1)

        Integration_shared.compute_delj(dx, MInt, VInt, L, delj, use_delj_trick)
        Integration_shared.compute_abc_nobc(dx, dfactor, delj, MInt, V, dt, L, a, b, c)
        for ii in range(0, L):
            r[ii] = phi[ii][jj] / dt

        if jj == 0 and Mfirst <= 0:
            b[0] += (0.5 / nu1 - Mfirst) * 2. / dx[0]
        if jj == M - 1 and Mlast >= 0:
            b[L - 1] += -(-0.5 / nu1 - Mlast) * 2. / dx[L - 2]

        Integration.tridiag(a, b, c, r, temp, L)
        for ii in range(0, L):
            phi[ii][jj] = temp[ii]


def implicit_2Dy(phi, xx, yy, nu2, m21, gamma2, h2, dt, use_delj_trick):
    L = phi.shape[0]
    M = phi.shape[1]
    dy = torch.diff(yy)
    dfactor = torch.zeros(M)
    yInt = torch.zeros(M - 1)

    Integration_shared.compute_dfactor(dy, M, dfactor)
    Integration_shared.compute_xInt(yy, M, yInt)

    MInt = torch.zeros(M - 1)
    V = torch.zeros(M)
    VInt = torch.zeros(M - 1)
    delj = torch.zeros(M - 1)

    for jj in range(0, M):
        V[jj] = Integration_shared.Vfunc(yy[jj], nu2)
    for jj in range(0, M - 1):
        VInt[jj] = Integration_shared.Vfunc(yInt[jj], nu2)

    a = torch.zeros(L)
    b = torch.zeros(L)
    c = torch.zeros(L)
    r = torch.zeros(L)

    for ii in range(0, L):
        x = xx[ii]
        Mfirst = Integration_shared.Mfunc2D(yy[0], x, m21, gamma2, h2)
        Mlast = Integration_shared.Mfunc2D(yy[M-1], x, m21, gamma2, h2)

        for jj in range(0, M-1):
            MInt[jj] = Integration_shared.Mfunc2D(yInt[jj], x, m21, gamma2, h2)

        Integration_shared.compute_delj(dy, MInt, VInt, M, delj, use_delj_trick)
        Integration_shared.compute_abc_nobc(dy, dfactor, delj, MInt, V, dt, M, a, b, c)

        for jj in range(0, M):
            r[jj] = phi[ii][jj] / dt

        if ii == 0 and Mfirst <= 0:
            b[0] += (0.5 / nu2 - Mfirst) * 2. / dy[0]
        if ii == L-1 and Mlast >= 0:
            b[M-1] += -(-0.5 / nu2 - Mlast) * 2. / dy[M-2]

        Integration.tridiag(a, b, c, r, phi[ii], M)


def implicit_precalc_2Dx(phi, ax, bx, cx, dt):
    L = phi.shape[0]
    M = phi.shape[1]
    a = torch.zeros(L)
    b = torch.zeros(L)
    c = torch.zeros(L)
    r = torch.zeros(L)
    temp = torch.zeros(L)
    for jj in range(0, M):
        for ii in range(0, L):
            a[ii] = ax[ii][jj]
            b[ii] = bx[ii][jj] + 1/dt
            c[ii] = cx[ii][jj]
            r[ii] = 1/dt * phi[ii][jj]

        Integration.tridiag(a, b, c, r, temp, L)
        for ii in range(0, L):
            phi[ii][jj] = temp[ii]


def implicit_precalc_2Dy(phi, ay, by, cy, dt):
    L = phi.shape[0]
    M = phi.shape[1]
    b = torch.zeros(M)
    r = torch.zeros(M)

    for ii in range(0, L):
        for jj in range(0, M):
            b[jj] = by[ii][jj] + 1/dt
            r[jj] = 1/dt * phi[ii][jj]

        Integration.tridiag(ay[ii], b, cy[ii], r, phi[ii], M)

