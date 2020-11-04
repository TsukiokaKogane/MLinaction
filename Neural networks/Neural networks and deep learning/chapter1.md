# Using neural nets to recognize handwritten digits
object: write a computer program implementing a neural network that learns to recognize handwritten digits.

## Perceptrons
- A perceptron takes several $\textbf{binary}$ inputs $ x_1,x_2,...$ and produces a single $\textbf{binary}$ output:
![perceptron](img/perceptron1.png)
In the example shown a perceptron has three $inputs$, $x_1, x_2, x_3$.
- $weights$ are real numbers expressing the importance of the respective inputs to the output.
- The neuron's $output$, 0 or 1, is determined by whether the weighted sum $\sum_jw_jx_j$  is less than or greater than some threshold value.

$$output=
\tag{1}
\begin{cases}
0 &  if & \sum_jw_jx_j \leq threshold\\ 
1 &  if & \sum_jw_jx_j > threshold
\end{cases}$$

$\textbf{intuition:}$ perceptron is a device that makes decisions by weighing up evidence.

simplified perceptrons
- $w \cdot x \equiv  \sum_jw_jx_j$
- $ b \equiv - threshold $  (perceptron's $bias$: a measure of how easy it is to get the perceptron to output 1)
$$output=
\tag{2}
\begin{cases}
0 &  if & w \cdot x + b \leq 0\\ 
1 &  if & w \cdot x + b > 0
\end{cases}$$

## Sigmoid neurons
$\textbf{intuition:}$ a small change in a weight (or bias) causes only a small change in output.
- sigmoid neuron has $inputs$, $x_1,x_2,...$ these inputs can take on $\textbf{any}$ values $between$ 0 and 1.
- the sigmoid neuron has $weights$ for each input,$w_1,w_2,...$,  and an overall $bias$, $b$.
- $output$ is $\sigma(w \cdot x + b)$, where $\sigma$ is called the sigmoid function[^1], defined by:
$$
\tag{3}
\sigma(z) \equiv \frac{1}{1 + e^{-z}}
$$

[^1]:$\sigma$ is sometimes called the logistic function, and this new class of neurons called logistic neurons.

By using the actual $\sigma$ function we get a $\textbf{smoothed}$ out perceptron. The smoothness of $\sigma$ means that small changes $\Delta w_j$ in the weights and $\Delta b$ in the bias will produce a small change $\Delta \text{output}$ in the output from the neuron.
$$
\tag{4}
\Delta \text{output} = \sum_j{ \frac{\partial \text{ output}}{\partial w_j} \Delta w_j + \frac{\partial \text{ output}}{\partial b} \Delta b} 
$$
$\textbf{intuition:}$ $\Delta \text{output}$ is a linear function of the changes $\Delta w_j$ and $\Delta b$ in the $weights$ and $bias$. This linearity makes it easy to choose small changes in the weights and biases to achieve any desired small change in the output. 

### Further discussion
- activation function: it's the $\textbf{shape}$ of $\sigma$ which really matters, and not its exact form. However, when we compute those partial derivatives in $(4)$, using $\sigma$ will simplify the algebra. In any case, $\sigma$ is commonly-used in work on neural nets, and is the activation function we'll use most often.

## The architecture of neural networks
- The leftmost layer in this network is called the input layer, and the neurons within the layer are called input neurons. 
- The rightmost or output layer contains the output neurons, or, as in this case, a single output neuron. 
- The middle layer is called a $hidden$ $layer$, since the neurons in this layer are neither inputs nor outputs.
![neuralnetwork](img/network1.png)
- The design of the input and output layers in a network is often straightforward.
-  neural networks researchers have developed many design heuristics for the $hidden$ $layers$, which help people get the behaviour they want out of their nets. For example, such heuristics can be used to help determine how to trade off the number of hidden layers against the time required to train the network.
- $feedforward$ neural networks: neural networks where the output from one layer is used as input to the next layer.
## A simple network to classify handwritten digits
split the problem of recognizing handwritten digits into two sub-problems:
- ($segmentation$ $problem$)：breaking an image containing many digits into a sequence of separate images, each containing a single digit.
- classifying individual digits 
To recognize individual digits we will use a three-layer neural network:
![neuralnetwork](img/network2.png)

## Learning with gradient descent
- notation $x$ to denote a training input.
- regard each training input $x$ as a $28×28=784$-dimensional vector
- denote the corresponding desired output by $y=y(x)$ , where $y$ is a $10$-dimensional vector. 
$\textbf{cost function}$[^2]:
$$
\tag{5}
C(w,b) = \frac{1}{2n}||y(x)-a||^2
$$
- $w$ denotes the collection of all weights in the network, 
- $b$ denotes all the biases, 
- $n$ is the total number of training inputs, a
- $a$ is the vector of outputs from the network when $x$ is input.

our $\textbf{goal}$ in training a neural network is to find weights and biases which minimize the quadratic cost function $C(w,b)$.

Calculus tells us that $C$ changes as follows:
$$
\tag{6}
\Delta C \approx \frac{\partial C}{\partial w}\Delta w + \frac{\partial C}{\partial b}\Delta b
$$

We denote the gradient vector by :
$$
\tag{7}
\nabla C \equiv \left(\frac{\partial C}{\partial w}, \frac{\partial C}{\partial b} \right)^T
$$
Let $\Delta v \equiv (\Delta w, \Delta b)$
With these definitions, the expression $(6)$ for $\Delta C$ an be rewritten as:
$$
\tag{8}
\Delta C \approx \nabla C \cdot \Delta v
$$
suppose we choose:
$$
\tag{9}
\Delta v = - \eta \nabla C
$$
where $\eta$ is a small, positive parameter (known as the $learning$ $rate$). Then Equation $(8)$ tells us that $\Delta C \approx \nabla C \cdot \Delta v = - \eta ||\nabla C||^2$, Because $||\nabla C||^2 \geq 0$, this guarantees that $\Delta C \leq 0$, so $C$ will always decrease.

gradient descent is the $\textbf{optimal}$ strategy for searching for a minimum:
constrain the size of the move so that $||\Delta v||=ϵ$ for some small fixed $ϵ>0$. The optimal strategy is equivalent to find the movement direction which decreases $C$ as much as possible. It can be proved that the choice of $\Delta v$ which minimizes $\nabla C \cdot \Delta v$ is $\Delta v = - \eta \nabla C$, where $\eta = \frac{\epsilon}{||\nabla C||}$ determined by the size constraint $||\Delta v|| = \epsilon$.
$\text{proof:}$ 
$$
||\nabla C \cdot \Delta v|| \leq  ||\nabla C||\cdot ||\Delta v|| = \epsilon ||\nabla C|| (\text{Cauchy-Schwarz inequality})\\ 
\text{when } \Delta v =  - \eta \nabla C \\
\nabla C \cdot \Delta v = - \eta ||\nabla C||^2 = - \epsilon ||\nabla C|| \\
$$

$\textbf{intuition:}$ gradient descent can be viewed as a way of taking small steps in the direction which does the most to immediately decrease $C$.

#### stochastic gradient descent:
$\textbf{intuition: }$ estimate the gradient $\nabla C$ by computing $\nabla C_x$ for a small sample of randomly chosen training inputs.
-  randomly picking out a small number $m$ of randomly chosen training inputs, label those random training inputs $X_1, X_2, ..., X_m$ , and refer to them as a mini-batch. 
- training with those
$$
\tag{10}
w_k \rightarrow w_k' = w_k - \frac{\eta}{m} \sum_j{\frac{\partial C_{X_j}}{\partial w_k}}
$$
$$
\tag{11}
b_l \rightarrow b_l' = b_l -  \frac{\eta}{m} \sum_j{\frac{\partial C_{X_j}}{\partial b_l}}
$$
where the sums are over all the training examples $X_j$ in the current mini-batch.
[^2]:Sometimes referred to as a loss or objective function.

