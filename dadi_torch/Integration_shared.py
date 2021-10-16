import torch


def Vfunc(x, nu):
    return 1./nu * x*(1.-x)


def Vfunc_beta(x, nu, beta):
    return 1./nu * x*(1.-x) * pow(beta+1, 2)/(4*beta)


def Mfunc1D(x, gamma, h):
    return gamma * 2*(h + (1.-2*h)*x) * x*(1.-x)


def Mfunc2D(x, y, m, gamma, h):
    return m * (y-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x)


def Mfunc3D(x, y, z, mxy, mxz, gamma, h):
    return mxy * (y-x) + mxz * (z-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x)


def Mfunc4D(x, y, z, a, mxy, mxz, mxa, gamma, h):
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x)


def Mfunc5D(x, y, z, a, b, mxy, mxz, mxa, mxb, gamma, h):
    return mxy * (y-x) + mxz * (z-x) + mxa * (a-x) + mxb * (b-x) + gamma * 2*(h + (1.-2*h)*x) * x*(1.-x)

# def compute_dx(xx, N, dx):
#     for ii in range(N-1):
#         dx[ii] = xx[ii+1]-xx[ii]


def compute_dfactor(dx, N, dfactor):
    for ii in range(1, N-1):
        dfactor[ii] = 2./(dx[ii] + dx[ii-1])
    dfactor[0] = 2./dx[0]
    dfactor[N-1] = 2./dx[N-2]


def compute_xInt(xx, N, xInt):
    for ii in range(0, N-1):
        xInt[ii] = 0.5*(xx[ii+1]+xx[ii])


def compute_delj(dx, MInt, VInt, delj, use_delj_trick):
    if not use_delj_trick:
        torch.fill_(delj, 0.5)
        return
    wj = 2 * MInt * dx
    epsj = torch.exp(wj / VInt)
    if not torch.equal(epsj, torch.full(epsj.size(), 1.0)) and not torch.equal(wj, torch.zeros(wj.size())):
        delj = (-epsj * wj + epsj * VInt - VInt) / (wj - epsj * wj)


def compute_abc_nobc(dx, dfactor, delj, MInt, V, N, a, b, c):
    # Using atemp and ctemp yields an ~10% speed-up
    for ii in range(0, N-1):
        atemp = MInt[ii] * delj[ii] + V[ii]/(2*dx[ii])
        a[ii+1] = -dfactor[ii+1]*atemp
        b[ii] += dfactor[ii]*atemp
        ctemp = -MInt[ii] * (1 - delj[ii]) + V[ii+1]/(2*dx[ii])
        b[ii+1] += dfactor[ii+1]*ctemp
        c[ii] = -dfactor[ii]*ctemp

