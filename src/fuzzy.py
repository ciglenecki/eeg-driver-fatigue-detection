def default(x, r):
    assert isinstance(
        r, tuple), 'When Fx = "Default", r must be a two-element tuple.'
    y = np.exp(-(x**r[1])/r[0])
    return y


def FuzzEn(Sig,  m=2, tau=1, r=(.2, 2), Fx='default', Logx=np.exp(1)):

    N = Sig.shape[0]

    m = m+1
    Fun = default
    Sx = np.zeros((N, m))
    for k in range(m):
        Sx[:N-k*tau, k] = Sig[k*tau::]

    Ps1 = np.zeros(m)
    Ps2 = np.zeros(m-1)
    Ps1[0] = .5
    for k in range(2, m+1):
        N1 = N - k*tau
        N2 = N - (k-1)*tau
        T2 = Sx[:N2, :k] - \
            np.transpose(np.tile(np.mean(Sx[:N2, :k], axis=1), (k, 1)))
        d2 = np.zeros((N2-1, N2-1))

        for p in range(N2-1):
            Mu2 = np.max(
                np.abs(np.tile(T2[p, :], (N2-p-1, 1)) - T2[p+1:, :]), axis=1)
            d2[p, p:N2] = Fun(Mu2, r)

        d1 = d2[:N1-1, :N1-1]
        Ps1[k-1] = np.sum(d1)/(N1*(N1-1))
        Ps2[k-2] = np.sum(d2)/(N2*(N2-1))

    with np.errstate(divide='ignore', invalid='ignore'):
        Fuzz = (np.log(Ps1[:-1]) - np.log(Ps2))/np.log(Logx)

    return Fuzz, Ps1, Ps2


def sigmoid(x, r):
    assert isinstance(
        r, tuple), 'When Fx = "Sigmoid", r must be a two-element tuple.'
    y = 1/(1 + np.exp((x-r[1])/r[0]))
    return y


def modsampen(x, r):
    assert isinstance(
        r, tuple), 'When Fx = "Modsampen", r must be a two-element tuple.'
    y = 1/(1 + np.exp((x-r[1])/r[0]))
    return y


def gudermannian(x, r):
    if r <= 0:
        raise Exception('When Fx = "Gudermannian", r must be a scalar > 0.')
    y = np.arctan(np.tanh(r/x))
    y = y/np.max(y)
    return y


def linear(x, r):
    if r == 0 and x.shape[0] > 1:
        y = np.exp(-(x - min(x))/np.ptp(x))
    elif r == 1:
        y = np.exp(-(x - min(x)))
    elif r == 0 and x.shape[0] == 1:
        y = 0
    else:
        print(r)
        raise Exception('When Fx = "Linear", r must be 0 or 1')
    return y
