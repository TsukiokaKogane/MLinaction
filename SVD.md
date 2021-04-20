<b>定义 1.1</b> 矩阵的奇异值分解是指将一个非零的$ m \times n$实矩阵 $A$, $A \in R^{ m \times n }$，表示为以下三个实矩阵乘积形式的运算，即进行矩阵的因子分解：
$$
A = U \Sigma V^T
\tag{1.1}
$$
其中$U$是$m$阶正交矩阵，$V$是$n$阶正交矩阵，$\Sigma$ 是由降序排列的非负的对角线元素组成的 $m \times n$ 矩形对角矩阵。满足
$$
UU^T = I \\
VV^T = I \\
\Sigma = diag(\sigma_1,\sigma_2, ..., \sigma_p) \\
\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_p \geq 0\\
p = \min(m, n)
$$
注意有:
$$
AV = U\Sigma
$$
$U\Sigma V^T$ 称为矩阵$A$的奇异值分解(singular value decomposition, SVD), $\sigma_i$称为矩阵$A$的奇异值， $U$的列向量称为左奇异向量，$V$的列向量称为右奇异向量。
矩阵的奇异值分解不是唯一的。
任意给定一个实矩阵，其奇异值分解一定存在。
其中$V=\{v_1,v_2,...,v_n\}$的列向量对应$A^TA$的特征向量，$U_1=\{u_1,u_2,...,u_r\}$的列向量$u_j=\frac{1}{\sigma_j}Av_j$。$U_2$为$N(A^T)$的一组标准正交基，$U=[U_1 U_2]$。

<b>定义 1.2</b> 设有 $m \times n$实矩阵A,其秩为$rank(A) = r, r\leq \min(m,n)$，则称$U_r\Sigma_rV_r^T$为$A$的紧奇异值分解(compact SVD)
$$
A = U_r\Sigma_rV_r^T
\tag{1.2}
$$
其中$U_r$是$m \times r$矩阵，$V_r$是$n \times r$矩阵，$\Sigma_r$是$r$阶对角矩阵；矩阵$U_r$由完全奇异值分解中$U$的前$r$列、矩阵$V_r$由$V$的前$r$列，矩阵$\Sigma_r$由$\Sigma$的前$r$个对角线元素得到。

<b>定义 1.3</b> 设有 $m \times n$实矩阵A,其秩为$rank(A) = r$,且$0 < k < r$,则称$U_k\Sigma_kV_k^T$为$A$的截断奇异值分解(truncated SVD)
$$
A \approx U_k\Sigma_kV_k^T
$$
其中$U_r$是$m \times k$矩阵，$V_k$是$n \times k$矩阵，$\Sigma_k$是$k$阶对角矩阵；矩阵$U_k$由完全奇异值分解中$U$的前$k$列、矩阵$V_k$由$V$的前$k$列，矩阵$\Sigma_k$由$\Sigma$的前$k$个对角线元素得到。
奇异值分解是在平方损失(弗罗贝尼乌斯范数)意义下对矩阵对最优近似。

### SVD应用
#### PCA
- 样本主成分分析：假设对$m$维随机变量 $ \boldsymbol{x} = (x_1,x_2,...,x_m)^T$ 进行$n$次独立观测，$\boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n$ 表示观测样本，其中$\boldsymbol{x}_j = (x_{1j},x_{2j},...,x_{mj})^T$表示第$j$个观测样本。观测数据用样本矩阵$\boldsymbol{X}$ 表示，记作：
$$
X_{m \times n} = [\boldsymbol{x}_1 \boldsymbol{x}_2 ... \boldsymbol{x}_n]
$$
对于规范化的样本矩阵，样本协方差矩阵$S$就是样本相关矩阵$R$
$$
R = \frac{1}{n-1}X X^T
$$
通过SVD, $X=U \Sigma V^T$，代入可得：
$$
R = \frac{1}{n-1} U \Sigma \Sigma U^T
$$
样本主成分分析可以转化为求样本相关矩阵R的$k$个特征$\lambda_i$和对应的单位特征向量$\alpha_i$。不妨令$A = [\alpha_1, \alpha_2, ..., \alpha_k]$, 即求：
$$
R_{m \times m} A_{m \times k} =  A_{m \times k} \Sigma_{k \times k}
$$
根据
$$
R = \frac{1}{n-1} U \Sigma \Sigma U^T \\
$$
$$
AV = U\Sigma
$$
此时，样本主成分分析等价于求$R$的右奇异向量$A_{m \times k}(V)$,等价于求$X$的左奇异向量$U^T$
#### 行数压缩
假设样本是$m \times n$的矩阵$A$， 如果通过SVD找到矩阵$AA^T$的最大$d$个特征向量张成的$m \times d$维矩阵$U$，则我们如果进行如下处理：
$$
A^\prime_{d \times n} = U^T_{d \times m} A_{m \times n}
$$
可以得到 $d \times n$的矩阵$A^\prime$，这个矩阵和我们原来的$m \times n$维样本矩阵$A$相比，行数从$m$减到了$d$，对行数进行了压缩。