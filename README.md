# Decision Tree (CART algorithm)

## 1. Tree Base Model:

Given N sample $(x_{i}, y_{i}), i \in [1, 2, ...N]$ with:
- Training vector, $x_{i} = (x_{i1}, x_{i2}, ..., x_{ip})$ with p is number of features.
- Label vector: $y \in \R^{N}$.

A Decision Tree recursively partitions the feature sapce such that samples with same labels or similar target values are grouped together. Tree splits the sample into M region $R_{1}, R_{2}, ..., R_{M}$

Each sample will have following prediction:
$$f(x) = \sum_{i}^{M} c_{m}I(x \in \R_{m})$$

For regression, $c_{m}$ will be the average of all samples in that leaf, if the loss metric is sum of squares

For classification, $c_{m}$ will be the majority class in that leaf.

The way Decision Tree splits the sample to build a tree is use a greedy algorithm to choose the best split. So at each split we search across all variable and all the value of this variable to find the best split. 

Let the data at node $m$ be represented by $Q_{m}$ with $N_{m}$ samples at that node. For each candidate split $\theta = (j, t_{m})$ consisting of a features $j$ and threshold $t_{m}$, partition the data into:
$$Q^{left}_{m}(\theta) = ((x, y)|x_{j} <= t_{m})$$

$$Q^{right}_{m}(\theta) = Q_{m} \ Q_{m}^{left}(\theta)$$

The quality of a candidate split of node $m$ is then computed using an impurity function or loss function $H()$, the choice of which depends on the task being solved(classification or regression)

$$G_{Q_{m}, \theta} = \frac{N_{m}^{left}}{N_{m}}H(Q^{left}_{m}(\theta)) + \frac{N_{m}^{right}}{N_{m}}H(Q^{right}_{m}(\theta))$$

Select the parameters that minimises the impurity

$$\theta^{*} = argmin_{\theta}G(Q_{m}, \theta)$$

Recurse for subsets $Q_{m}^{left}(\theta^{*})$ and $Q_{m}^{right}(\theta^{*})$ until the maximum allowable depth is reached.


## 2. Classification Tree

***For classification problem*** when target value has $K$ classes, at leaf m, the probability of the majority class will be:
$$p_{mk} = \frac{1}{N_{m}} \sum_{x_{i} \in R_{m}} I(y_{i} = k)$$

Using this, we have common measures of impurity are the following:

- Gini(defaul):
  $$H(Q_{m}) = \sum_{k}p_{mk}(1 - p_{mk})$$
- Entropy:
  $$H(Q_{m}) = - \sum_{k}p_{mk}log_{2}(p_{mk})$$

So we iterate across all the variables and their value to find the optimal split, which find $\theta^{*}$ to minimize $G(Q_{m}, \theta)$
## 3. Regression Tree

If the target is a continuous value, then for node m, common criteria to minimize as for determining locations for future splits are:

- Min Squared Error(MSE)
  $$\overline{y}_{m} = \frac{1}{N_{m}} \sum_{y\in Q_{m}}y$$
  $$H(Q_{m}) = \frac{1}{N_{m}}\sum_{y\in Q_{m}}(y - \overline{y}_{m})^{2}$$

Note: In order to compute MSE in O(#sample) we decompose $\sum_{y\in Q_{m}}(y - \overline{y}_{m})^{2}$ into $\sum(y^{2}) - n\overline{y}^{2}$

- Or Mean Absoluted Error(MAE) (Much slower than MSE)
  $$median(y)_{m} = \underset{y\in Q_{m}}{median(y)}$$
  $$H(Q_{m}) = \frac{1}{N_{m}}|y - median(y)_{m}|$$


## 4. Pruning

We found the best split so we can build a tree from top to the bottom. But if we built a very large tree then it might overfit the data, while a small tree might not capture the importance structure(underfit).

Tree size is a tuning parameter governing the model's complexity, and the optimal tree size should be adaptively chosen from the data.

One approach is set some parameters and just split tree nodes only if is those parameters does not exceed some threshold (**Early Stopping**).

How ever, this strategy can be too short-sighted because sometime a seemingly worthless split might lead to a very good split below it. So we can use **Post Pruning**

### 4.1 Early Stopping (pre-pruning)

With early stopping, following hyper parameters are used:

|**parameter** | **Description**|
| :---:        | :---          |
|*min_samples_split* | Minimum number of samples in an internal node |
|*min_samples_leaf*  | Minimum number of samples in a leaf |
|*max_depth*   | Maximal tree depth (a key parameter for model, especially when input features are very big) |
|*min_impurity_split* | When impurity < threshold then stop growing |

### 4.2 Post Pruning 

Another strategy is to grow a large tree $T$, stopping the splitting process only when some minimum node size is reached. Then this large tree is pruned using ***minimal cost-complexity pruning***  algorithm. This algorithm is parameterized by $\alpha > 0$ known as the complexity parameter. The complexity parameter is used to define the cost-complexity measure, $R_{\alpha}(T)$ of a given tree T:
$$R_{\alpha}(T) = R(T) + \alpha |T|$$

Where $|T|$ is the number of terminal nodes in T and $R(T)$ is total sample weighted impurity of the terminal nodes. As shown above, the impurity of a node depends on the criterion. Minimal cost-complexity pruning finds the subtree of $T$ that minimizes $R_{\alpha}(T)$.

The cost comlexity measure of a single node is $R_{\alpha}(T) = R(t) + \alpha$. 

The branch, $T_{t}$, is defined to be a tree where node $t$ is its root. In general, the impurity of a node is greater than the sum of impurities of its terminal nodes, $R(T_{t}) < R(t)$ 

However, the cost complexity measure of a node, $t$, and its branch, $T_{t}$, can be equal depending on $\alpha$. We define the effective $\alpha$ of a node to be the value where they are equal, $R_{\alpha}(T_{t}) = R_{\alpha}(t)$ or $\alpha_{eff}(t) = \frac{R(t) - R(T_{t})}{|T|-1}$.

A non-terminal node with the smallest value of $\alpha_{eff}$ is the weakest link and will be pruned. This process stops when the pruned tree's minimal $\alpha_{eff}$ is greater than the **cpp_alpha parameter**.