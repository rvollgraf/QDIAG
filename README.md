# QDIAG - Quadratic Optimization for Simultaneous Matrix Diagonalization

This is a PyTorch implementation of the joint matrix diagonalization algorithm proposed in the paper

Vollgraf, Roland, and Klaus Obermayer. "Quadratic optimization for simultaneous matrix diagonalization." *IEEE Transactions on Signal Processing 54.9* (2006): 3270-3278. (https://ieeexplore.ieee.org/abstract/document/1677895)

Run a simple test script with
```bash
python -m qdiag
```




```python
W = qdiag( C0, C) 
``` 
finds a matrix W so that `W.T @ C0 @ W` has diagonal elements
equal to 1 and `W.T @ C[i] @ W` has smallest possible off-diagonal
elements. C0 is a positive definite NxN matrix, and C is a KxNxN array
of K correlation matrices.

```python
W = qdiag( C0, C, p)
``` 
where p is a vector with K positive elements,
weights the matrices in C according to them. An empty vector has the
effect that all matrices are weighted equally.

```python
qdiag( C0, C, p, **kwargs)
```
allows to pass optional arguments. These 
can be the following  (default values are taken for missing fields):

|key|value|
|---|---|
| `verbose`  | print informational messages in every iteration (default: false) |
| `approach` | implementation of QDIAG, O(KN^3) or O(N^5), see [1]. Possible  values  `'OKN3'`, `'ON5'` (default: `'OKN3'`) |
| `Nit`      | maximum number of iteration (default 100) |
|  `tol_w`   | Stopping criterion. Changes of w smaller than this value terminate the iterations (default: 1e-4) |
| `W`         | used as the inital W (default: random NxN matrix)
|  `M`       | number components, columns of W (default: M=N  matrix) opt.W has priority over opt.M
| `return_errlog` | return also a (Nit) tensor of diagonalization error history (default: False)
| `return_Wlog` | return a  (Nit,N,M) tensor of diagonalization matrix history (default: False)




The code is derived from https://www.ni.tu-berlin.de/menue/software_and_data/approximate_simultaneous_matrix_diagonalization_qdiag
