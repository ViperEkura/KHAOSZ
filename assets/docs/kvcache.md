## kv_cache 实现

根据注意力的计算公式

$$
\begin{align*}
o_i &= \sum_j s_{ij} v_{j} \newline
s_{ij} &= \text{softmax}\left( \frac{q_{i} k_{j}}{\sqrt{d_k}} \right)
\end{align*}
$$

由于模型是自回归模型, 我们只用求序列最后一个部分，也就是说 $ i $ 的下标是确定的, 是序列最后一个元素, 我们求的是 $o_{n} $ 

$$
\begin{align*}
o_n &= \sum_j s_{j}v_{j} \newline
s_j &= \text{softmax}\left(\frac{q_n k_{j}}{\sqrt{d_k}} \right)
\end{align*}
$$

如果我们把式子展开

$$
o_n = \sum_j \text{softmax}\left(\frac{q_n k_{j}}{\sqrt{d_k}}\right)v_{j}
$$

以上表达式只有k和v存在长度下标, 而 $q$ 没有， 所以计算过程中 $q$ 的输入是确定的上次输入的最后一个token, 而 $k,  v$ 是需要对不同长度的部分进行缓存的，同时缓存的时候应该注意位置编码的计算应该在kvcache的计算之前进行，否则会存在位置编码的计算错误