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
### EM算法
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
\theta^{(i + 1)} = \arg \max _\theta Q(\theta, \theta^{(i)})
$$ 
(4) 重复(2)、(3)步直到收敛
