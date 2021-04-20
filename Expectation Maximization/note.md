# EM算法
$\textbf{intuition: }$有隐变量的概率模型参数的极大似然估计法

$\theta$为模型参数，将观测数据表示为$Y=\left(Y_1, Y_2,...,Y_n \right)^T$，未观测数据表示为$Z=\left(Z_1,Z_2,...,Z_n\right)^T$，则观测数据的似然函数为：
$$
\tag{1}
P(Y|\theta) = \sum_Z{P(Z|\theta)P(Y|Z,\theta)}
$$
考虑求模型参数$\theta$的极大似然估计,即:
$$
\tag{2}
\hat{\theta} = \arg \max_\theta \log P(Y|\theta)
$$

$Y$ 和 $Z$ 连在一起称为完全数据，观测数据 $Y$ 又称为不完全数据。 
假设给定观测数据 $Y$, 其概率分布是 $P(Y|\theta)$ ，其中 $\theta$ 是需要估计的模型参数，那么不完全数据 $Y$ 的似然函数是 $\hat{P}(Y|\theta)$ , 对数似然函数 $L(\theta) = \log \hat{P}(Y|\theta)$；假设 $Y$ 和 $Z$ 的联合概率分布是 $P(Y,Z|\theta)$, 那么完全数据的对数似然函数是 $L(\theta) = \log P(Y,Z|\theta)$ 
## EM算法流程
输入:观测变量数据 $Y$ ,隐变量数据 $Z$ ，联合分布$P(Y,Z|\theta)$，条件分布$P(Z|Y,\theta)$；
输出：模型参数 $\theta$
(1) 选择参数初值 $\theta^{(0)}$ ，开始迭代；
(2) E步：计 $\theta^{(i)}$ 为第 $i$ 次迭代参数 $\theta$ 的估计值，在第 $i+1$ 次迭代的E步，计算
$$
\tag{3}
\begin{aligned}
Q(\theta,\theta^{(i)}) & = E_Z[\log P(Y,Z|\theta)|y, \theta^{(i)}] \\
                       & = E_{Z} \log P(Y,Z|\theta) P(Z|Y, \theta^{(i)})
\end{aligned}
$$

$Q$ 函数是完全数据的对数似然函数 $\log P(Y, Z|\theta)$ 关于给定观测数据 $Y$ 和当前参数 $\theta^{(i)}$ 下对未观测数据 $Z$ 的条件概率分布 $P(Z|Y, \theta^{(i)})$ 的期望

(3) M步：求使 $Q(\theta, \theta^{(i)})$ 极大化的 $\theta$ ，确定第 $i + 1 $ 次迭代的参数的估计值 $\theta^{(i + 1)} $
$$
\tag{4}
\theta^{(i + 1)} = \arg \max _\theta Q(\theta, \theta^{(i)})
$$ 
(4) 重复(2)、(3)步直到收敛

## EM算法的导出
优化目标: 
$$
\tag{5}
 \arg \max_\theta L(\theta)
$$
假设在第 $i$ 次迭代后 $\theta$ 的估计值是 $\theta^{(i)}$ ，考虑新估计值 $L(\theta) > L(\theta^{(i)})$。
$$
\tag{6}
\begin{aligned}
L(\theta) - L(\theta^{(i)}) & = \log \left(\sum_ZP(Y|Z, \theta) P(Z|\theta) \right) - \log P(Y|\theta^{(i)}) \\
& = \log \left( \sum_Z P(Z|Y, \theta^{(i)})\frac{ P(Y|Z, \theta) P(Z|\theta)}{P(Z|Y, \theta^{(i)})} \right) - \log P(Y|\theta^{(i)}) \\
& \geq \sum_Z P(Z|Y, \theta^{(i)}) \log \frac{ P(Y|Z, \theta) P(Z|\theta)}{P(Z|Y, \theta^{(i)})} - \log P(Y|\theta^{(i)}) \quad (Jensen 不等式)\\
& = \sum_Z P(Z|Y, \theta^{(i)}) \log \frac{ P(Y|Z, \theta) P(Z|\theta)}{P(Z|Y, \theta^{(i)}) \log P(Y|\theta^{(i)})}
\end{aligned}
$$
考虑下一次迭代参数 $\theta^{(i+1)}$
$$
\tag{7}
\begin{aligned}
\theta^{(i+1)} &= \arg \max_\theta \left(L(\theta^{(i)}) +  \sum_Z P(Z|Y, \theta^{(i)}) \log \frac{ P(Y|Z, \theta) P(Z|\theta)}{P(Z|Y, \theta^{(i)}) \log P(Y|\theta^{(i)})} \right) \\
&=\arg \max_\theta \sum_Z P(Z|Y, \theta^{(i)}) \log  P(Y|Z, \theta) P(Z|\theta) \\
&=\arg \max_\theta \sum_Z P(Z|Y, \theta^{(i)}) \log  P(Y,Z |\theta) 
\end{aligned}
$$

## HMM中的应用（Baum-Welch算法）
### HMM的定义：
HMM 由初始概率分布、状态转移概率以及观测概率确定。
设 $Q$ 是所有可能的状态的集合， $V$ 是所有可能的观测的集合：
$$
Q = \{q_1,q_2,...,q_N\} \quad V=\{v_1,v_2,..,v_M\}
$$
其中， $N$ 是可能的状态数，$M$ 是可能的观测数。
$I$ 是长度为 $T$ 的状态序列， $O$是对应的观测序列：
$$
I = (i_1,i_2,...,i_T) \quad O=(o_1,o_2,..,o_T)
$$
$A$ 是状态转移矩阵:
$$A = [a_{ij}]_{N \times N}$$
其中，
$$
a_{ij} = P(i_{t+1} = q_j | i_t = q_i) \quad i =1,2,...,N; \quad j =1,2,...,N
$$
是在时刻t处于状态$q_i$的条件下在时刻 $t+1$ 转移到状态 $q_j$ 的概率。
$B$ 是观测概率矩阵：
$$
B = [b_j(k)]_{N \times M}
$$
其中，
$$
b_j(k) = P(o_t=v_k | i_t = q_j) \quad k=1,2,...,M; \quad j=1,2,...,N 
$$
是在时刻 $t$ 处于状态 $q_j$ 的条件下生成观测 $v_k$ 的概率。
$\pi$ 是初始状态概率向量：
$$\pi = (\pi_i)$$
其中，
$$\pi_i = P(i_1 = q_i), \quad i=1,2,..,N$$
是时刻 $t=1$ 时处于状态 $q_i$ 的概率。

HMM基本假设：
(1) 齐次马尔可夫性假设：HMM在任意时刻 $t$ 的状态只依赖于其前一时刻的状态
$$P(i_t | i_{t-1}, o_{t-1},...) = P(i_t|i_{t-1}) t =1,2,...,T
$$ 
(2) 观测独立性假设，任意时刻的观测只依赖于该时刻HMM的状态
$$P(o_t|i_t,o_{t-1},...) = P(o_t,|i_t)$$

### Baum-Welch算法
根据HMM的定义，令$\lambda = (\pi, A, B)$ ，可以得到 $Q$ 函数：
$$
\begin{aligned}
Q(\lambda , \bar{\lambda}) & = \sum_I \log P(O,I | \lambda) P(I | O, \bar{\lambda}) \\
& =
\end{aligned}
$$