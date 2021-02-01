import torch
from torch import Tensor


# noinspection PyShadowingNames
def err(W: torch.Tensor, C0: torch.Tensor, C: torch.Tensor, p: torch.Tensor) -> float:
    """
    err computes joint-diagonalization error
       E = err(W, C0, C, p);
       E(1) error according to QDIAG
    """
    M = W.shape[1]

    d = (W.T @ C0 @ W).diag().unsqueeze(0)
    W = W / d.sqrt()

    D = (C.transpose(1, 2) @ W).transpose(1, 2) @ W
    D = D ** 2

    E = p @ (D.sum(dim=(1, 2)) - D.diagonal(dim1=1, dim2=2).sum(dim=1))

    E = E / (M ** 2 - M)

    return E.item()


# noinspection PyUnboundLocalVariable,PyShadowingNames
def qdiag(C0: torch.Tensor,
          C: torch.Tensor,
          p: torch.Tensor = None,
          verbose: bool = False,
          approach: str = 'OKN3',
          Nit: int = 100,
          tol_w: float = 1e-6,
          W: torch.tensor = None,
          M: int = None,
          return_errlog: bool = False,
          return_Wlog: bool = False
          ):
    """
    QDIAG joint matrix diagonalization

    W = qdiag( C0, C) finds a matrix W so that W'*C0*W has diagonal elements
    equal to 1 and W'*C(:,:,i)*W has smallest possible off-diagonal
    elements. C0 is a positive definite NxN matrix, and C is a NxNxK array
    of K correlation matrices.

    W = qdiag( C0, C, p), where p is a vector with K positive elements,
    weights the matrices in C according to them. An empty vector has the
    effect that all matrices are weighted equally.

    qdiag( C0, C, p, opt) allows to pass an option structure. opt can have
    the following fields (default values are taken for missing fields):

    'verbose'  : print informational messages in every iteration (default:
                 false)
    'approach' : implementation of QDIAG, O(KN^3) or O(N^5), see [1].
                 Possible  values  'OKN3','ON5' (default: 'OKN3'):
    'Nit'      : maximum number of iteration (default 100)
    'tol_w'    : Stopping criterion. Changes of w smaller than this value
                 terminate the iterations (default: 1e-4)
    'W'        : used as the inital W (default: random NxN matrix)
    'M'        : number components, columns of W (default: M=N
                 matrix) opt.W has priority over opt.M



    [W, errlog] = qdiag( C0, C, p, opt) additionally outputs the
    diagonalization error in every iteration.

    [W, errlog, Wlog] = qdiag( C0, C, p, opt) additionally outputs the
    elements of W in every iteration.


    the algorithm is described in length in:

    [1] R. Vollgraf and K. Obermayer, Quadratic Optimization for Approximate
    Matrix Diagonalization, IEEE Transaction on Signal Processing, 2006, in
    press
    """

    assert approach in {'OKN3', 'ON5'}
    assert Nit > 1
    assert tol_w > 0

    assert C0.ndim == 2
    N: int = C0.shape[0]
    assert C0.shape == (N, N)
    assert torch.all(torch.eq(C0, C0.T))

    dtype = C0.dtype
    device = C0.device

    assert C.ndim == 3
    T: int = C.shape[0]
    assert C.shape == (T, N, N)

    tparams = dict(dtype=dtype, device=device)

    if M is None:
        M = N

    if W is None:
        W = torch.randn(N, M, **tparams)
    else:
        assert W.shape == (N, M)

    if p is None:
        p = torch.ones(T, **tparams) / T
    else:
        assert p.shape == (T,)
        p = p / p.sum()

    if return_errlog:
        err_log = torch.zeros(Nit, **tparams)

    if return_Wlog:
        W_log = torch.zeros(Nit, N, M, **tparams)

    def pavg(C_):
        return (p.view(T, 1, 1) * C_).sum(0)

    if verbose:
        print('Diagonalizing {:d} {:d}x{:d} matrices for {:d} components\nUsing approach {:s}\n'
              .format(T, N, N, M, approach))

    # normalize the length
    d = torch.diag(W.T @ C0 @ W)
    W = W / torch.sqrt(d).unsqueeze(0)

    # do the sphering of the whole problem wrt. C0
    db, vb = torch.symeig(C0, eigenvectors=True)
    assert torch.all(torch.gt(db, 0)), "C0 must be positive definite (all eigenvalues >0)"

    P = vb.T / torch.sqrt(db).unsqueeze(1)
    P_not = vb * torch.sqrt(db).unsqueeze(0)

    C = (C.transpose(1, 2) @ P.T).transpose(1, 2) @ P.T

    use_ON5 = approach == 'ON5'

    C0 = P @ C0 @ P.T
    W = P_not.T @ W

    # Initialization for O(N ^ 5)
    if use_ON5:
        C1 = C.view(T, N ** 2, 1)
        C2 = C.transpose(1, 2).reshape(T, N ** 2, 1)
        CC = pavg(C1 * C1.transpose(1, 2) + C2 * C2.transpose(1, 2))

        CC = CC.view(N, N, N, N)
        CC = CC.permute([2, 0, 3, 1])
        CC = CC.reshape(N * N, N * N)

        Om = W @ W.T

    # Initializations for O(KN^3)
    else:
        M1 = C @ W
        M2 = C.transpose(1, 2) @ W

        D = pavg(torch.bmm(M1, M1.transpose(1, 2)) + torch.bmm(M2, M2.transpose(1, 2)))

    # The iteration main loop
    for n in range(Nit):

        delta_w = 0

        # The inner loop over all columns of W
        for i in range(M):
            w = W[:, i].clone()

            # O(N^5 #####################
            if use_ON5:
                Om = Om - w.unsqueeze(0) * w.unsqueeze(1)
                D = (CC @ Om.flatten()).view(N, N)

            # O(KN^3) ###################
            else:
                m1 = C @ w
                m2 = C.transpose(1, 2) @ w
                D = D - pavg(m1.unsqueeze(1) * m1.unsqueeze(2) + m2.unsqueeze(1) * m2.unsqueeze(2))

            d, w_new = torch.symeig(D, eigenvectors=True)

            l: Tensor = torch.argmin(d)
            w_new = w_new[:, l]

            # compute the change in w. Note: w_new can have opposite sign
            delta_w = max(delta_w, min((w - w_new).norm(), (w + w_new).norm()))
            W[:, i] = w_new

            # O(N^5 #####################
            if use_ON5:
                Om = Om + w_new.unsqueeze(0) * w_new.unsqueeze(1)

            # O(KN^3) ###################
            else:
                m1 = C @ w_new
                m2 = C.transpose(1, 2) @ w_new
                D = D + pavg(m1.unsqueeze(1) * m1.unsqueeze(2) + m2.unsqueeze(1) * m2.unsqueeze(2))

            # end of inner loop

        if return_errlog:
            err_log[n] = err(W, C0, C, p)

        if return_Wlog:
            if n > 0:
                # invert some columns of W according to the previous one
                W_ = W * torch.sign((W_log[n - 1, :] * W).sum(dim=0, keepdim=True))
                W_log[n] = W_
            else:
                W_log[n] = W

        if verbose:
            print("It. {:4d}, diag. error: {:.06f}, max. change in W: {:6f}"
                  .format(n, err(W, C0, C, p), delta_w))

        if delta_w < tol_w:  # changes in W ar1e too small
            if verbose:
                print("Changes of W are smaller than {:f}".format(tol_w))
            break

        # end main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # revert the sphering
    W = P.T @ W

    ret = W,
    if return_errlog:
        ret += err_log[:n],
    if return_Wlog:
        ret += W_log[:n],

    return ret if len(ret) > 1 else ret[0]


if __name__ == "__main__":
    from matplotlib.pyplot import *

    # noinspection PyShadowingNames
    def run_test(T, N, M, method, symm='general'):

        # create positive definite C0 that defines the metric for W
        C0 = torch.randn(N, N)
        C0 = C0 @ C0.T + 10*torch.eye(N)

        # creat T matrices to be diagonalized, C0 being one of them
        C = torch.cat((0.1*C0.unsqueeze(0), torch.randn(T-1, N, N)), 0)

        if symm == 'pos_def':
            C = torch.bmm(C, C.transpose(1, 2))
        elif symm == 'symmetric':
            C = 0.5*(C+C.transpose(1, 2))

        p = torch.ones(T, dtype=C.dtype)

        W, err_log, W_log = qdiag(C0, C, p, verbose=True, approach=method, Nit=500, M=M,
                                  return_errlog=True, return_Wlog=True)

        D = (C.transpose(1, 2) @ W).transpose(1, 2) @ W

        imshow(W.T @ C0 @ W)
        show()

        figure()
        subplot(221)
        semilogy(err_log)
        subplot(222)
        plot(W_log.view(-1, N * M))
        subplot(212)
        imshow(D.permute(1, 0, 2).reshape(M, T * M).cpu().numpy())

        suptitle(f"{method} Digonalization of {T} {N}x{N} {symm} matrices in {M} directions ")
        show()

    for method_ in ['OKN3', 'ON5']:
        run_test(10, 25, 12, method_, 'symmetric')
        run_test(10, 25, 12, method_, 'pos_def')
        run_test(10, 25, 12, method_, 'general')
