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

### PCA Workflow

The implementation of PCA as a feature extraction method consists of a structured five-step sequence. The process begins with **Standardization**, where the dataset is scaled by first substracting the mean value (standardized values have a zero mean), then it dividing the result by the standard deviation (standardized values have a standard deviation equal to 1). Standardization is followed by the **Covariance Matrix Computation**, which captures the relationships between multiple variables in a dataset. 

Once the covariance matrix is established, the next phase involves **computing the eigenvectors and eigenvalues** of that matrix. This step is critical for identifying the principal components, which are directions in the feature space that represent the most variance. These components are typically ranked by their eigenvalues:
```math
\lambda_1 > \lambda_2 > \lambda_3 > \dots > \lambda_n
```

A **Feature Vector** is then constructed by selecting the eigenvectors corresponding to the largest eigenvalues. In the final step, the algorithm **recast the data along the principal component axes**. Geometrically, this transforms the original data, which may have been distributed across arbitrary feature axes, into a new coordinate system. 

<img width="1390" height="590" alt="image" src="https://github.com/user-attachments/assets/0d7dffe0-529b-439f-b377-e55c759ea3b1" />

A two-class dataset transformed through the PCA pipeline: Regarding the code implementation, the process begins with standardization using the `StandardScaler` component from the scikit-learn library. Subsequently, the **covariance matrix** is computed, and **the eigenvectors and eigenvalues** are derived using the `numpy.linalg.eig` function. To ensure that the most significant directions of variance are prioritized, the eigenvectors are sorted in descending order based on their corresponding eigenvalues. The final stage of the workflow involves **projecting the standardized data** onto the new principal component space. This **Projected Data' represents the original observations transformed into a new coordinate system where the axes are the principal components themselves. This is achieved using a **dot product** between the standardized matrix and the sorted eigenvector matrix.

## Linear Discriminant Analysis (LDA)

LDA is a powerful linear method used for dimensionality reduction. It specifically focuses on using class labels to find a new space where different groups are separated as clearly as possible. Its goal is to find linear combinations of features that best separate different classes of data. 

The LDA algorithm relies on two mathematical components: the **between-class scatter matrix** and the **within-class scatter matrix**. The process works by maximizing the scatter (distance) between different groups while minimizing the scatter (spread) inside each individual group. 
