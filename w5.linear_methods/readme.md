## Principal Component Analysis (PCA)

In the PCA method, the eigenvalues and eigenvectors of the covariance matrix, which represent the relationship between the data's feature vectors, are calculated. These eigenvectors are identified as the principal components. When PCA is applied for dimensionality reduction, the eigenvectors associated with small eigenvalues are removed. If necessary, the data can be transformed back to its original dimension. This process is also utilized during the feature extraction and feature selection stages.

Basic Relations of PCA

**Mean**
```math
\bar{X} = \frac{\sum_{i=1}^{n} X_i}{n}
```

**Variance**
```math
var(X) = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(X_i - \bar{X})}{(n - 1)}
```

**Covariance**
```math
cov(X, Y) = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{(n - 1)}
```

**Covariance Matrix**
```math
C = \begin{pmatrix} 
cov(x,x) & cov(x,y) & cov(x,z) \\ 
cov(y,x) & cov(y,y) & cov(y,z) \\ 
cov(z,x) & cov(z,y) & cov(z,z) 
\end{pmatrix}
```

### Eigenvalues and Eigenvectors

The **eigenvector** is a vector x that does not change its direction when multiplied by a matrix A. While most vectors rotate or shift when transformed by a matrix, an eigenvector remains on the same span.

The **eigenvalue** (λ) is the scalar value that represents how much the eigenvector is scaled during this transformation. It indicates whether the resulting vector Ax has been stretched, compressed, or kept at the same length compared to the original vector x.

The mathematical relationship:
```math
Ax = \lambda x
```

### Linear Transformation and Eigen-Decomposition

A linear transformation is defined by the action of a square matrix A on a vector x, resulting in a transformed vector b. This relationship serves as the foundation for mapping data from one space to another.

```math
Ax = b
```

In the context of eigen-decomposition, we search for specific vectors that maintain their orientation during this transformation. When a matrix A acts upon an eigenvector x, the output is a vector that lies on the same span as the original, scaled only by a factor known as the eigenvalue λ.

```math
Ax = \lambda x
```

For a transformation occurring in a three-dimensional space, a 3x3 matrix operates on a 3x1 column vector to produce a new 3x1 coordinate.

```math
\begin{bmatrix} 
a_{11} & a_{12} & a_{13} \\ 
a_{21} & a_{22} & a_{23} \\ 
a_{31} & a_{32} & a_{33} 
\end{bmatrix} 
\begin{bmatrix} 
x_1 \\ 
x_2 \\ 
x_3 
\end{bmatrix} 
= 
\begin{bmatrix} 
(a_{11} \cdot x_1) + (a_{12} \cdot x_2) + (a_{13} \cdot x_3) \\ 
(a_{21} \cdot x_1) + (a_{22} \cdot x_2) + (a_{23} \cdot x_3) \\ 
(a_{31} \cdot x_1) + (a_{32} \cdot x_2) + (a_{33} \cdot x_3) 
\end{bmatrix} 
= 
\begin{bmatrix} 
b_1 \\ 
b_2 \\ 
b_3 
\end{bmatrix}
```

In the specific case where x is an eigenvector, this results is equivalent to scaling the original vector by the eigenvalue λ.
```math
\begin{bmatrix} 
b_1 \\ 
b_2 \\ 
b_3 
\end{bmatrix} 
= 
\lambda 
\begin{bmatrix} 
x_1 \\ 
x_2 \\ 
x_3 
\end{bmatrix} 
= 
\begin{bmatrix} 
\lambda x_1 \\ 
\lambda x_2 \\ 
\lambda x_3 
\end{bmatrix}
```

<img width="419" height="328" alt="image" src="https://github.com/user-attachments/assets/e139c5ba-ca3c-4f16-840d-e9e4e70a6d1d" />

### Eigenvalue and Eigenvector Calculations

To determine the eigenvalues of a square matrix A, we identify the scalar values λ for which the matrix (A - λI) is singular, meaning its determinant is equal to zero. This leads to the characteristic equation:
```math
\det(A - \lambda I) = 0
```

Consider the following example with a 2 x 2 matrix A:
```math
A = \begin{bmatrix} 0.8 & 0.3 \\ 0.2 & 0.7 \end{bmatrix}
```

To find the eigenvalues, we first construct the shifted matrix by subtracting the product of the scalar λ and the identity matrix I from matrix A. This operation isolates the diagonal elements for the characteristic equation:
```math
A - \lambda I = \begin{bmatrix} 0.8 & 0.3 \\ 0.2 & 0.7 \end{bmatrix} - \begin{bmatrix} \lambda & 0 \\ 0 & \lambda \end{bmatrix} = \begin{bmatrix} 0.8 - \lambda & 0.3 \\ 0.2 & 0.7 - \lambda \end{bmatrix}
```
Next, we calculate the determinant of the resulting matrix. Setting this determinant to zero produces the characteristic quadratic equation:
```math
\det \begin{bmatrix} 0.8 - \lambda & 0.3 \\ 0.2 & 0.7 - \lambda \end{bmatrix} = (0.8 - \lambda)(0.7 - \lambda) - (0.3 \cdot 0.2) = \lambda^2 - \frac{3}{2}\lambda + \frac{1}{2} = 0
```

Factoring the resulting quadratic equation `(λ - 1)(λ - 0.5) = 0` yields two eigenvalues λ1 = 1 and λ2 = 0.5.

Once the eigenvalues are found, the corresponding eigenvectors x are determined by solving the linear system for each λ:
```math
(A - \lambda I)x = 0 \quad \text{or} \quad Ax = \lambda x
```

For the first eigenvalue λ1, the first eigenvector x1 is found to be:
```math
x_1 = \begin{bmatrix} 0.6 \\ 0.4 \end{bmatrix} \implies Ax_1 = \begin{bmatrix} 0.8 & 0.3 \\ 0.2 & 0.7 \end{bmatrix} \begin{bmatrix} 0.6 \\ 0.4 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.4 \end{bmatrix} = x_1
```

For the second eigenvalue λ2, the second eigenvector x2 is:
```math
x_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix} \implies Ax_2 = \begin{bmatrix} 0.8 & 0.3 \\ 0.2 & 0.7 \end{bmatrix} \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix} = \frac{1}{2}x_2
```

The logic:
1. Set up the characteristic equation `det(A - λI) = 0`;
2. Solve the resulting quadratic equation to find the eigenvalues (λ);
3. Plug each λ back into the system `(A - λI)*x = 0` to find the corresponding eigenvectors (x).

