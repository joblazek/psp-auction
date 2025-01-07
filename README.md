# psp-auction

Progressive Allocation in the PSP Auction follows this scheme:


% SELLER ALGORITHM
\begin{center}
\begin{algorithm}[H]
\caption{(Seller progressive allocation)}
\begin{algorithmic}[1]
\State $p^{j(0)} \gets \epsilon$, $s^{j(0)} \gets (p^j, D^j)$, $\bar\mcI =
\emptyset$, compute
$\mcI^{j(0)}$
\State Update $s^j$ 
\While{$D^j(t) > 0$}
\State $\bar{i} \gets \displaystyle\max_{i\in I^j}\sum_{i\in I^j} p_i^j$ 
\State $D^{j(t+1)} \gets D^{j(t)} - \g_{\bar{i}}^{j(t)}(a)$
%\State $p^j \gets \theta_{i^*}'(d_{i^*}^j)\circ e_i$
\State $p^j \gets p_{i^*}^j+\epsilon$ and $d^j \gets D^{j(t+1)}$
\State $s^{j(t+1)} \gets (d^j, p^j)$
\State Update $s^j$
\State $\bar\mcI \gets \bar\mcI \cup \bar{i}$
\For{$k \in \bar\mcI$}
\If{$p_k^j < p_{i*}^j$}
\State $D^{j(t+1)} = d_k^{j}$
\State $\bar\mcI \gets \bar\mcI \setminus \lbrace k \rbrace$
\EndIf
\EndFor
\State Compute $\mcI^{j(t)}$
\State $\mcI^{j(t+1)} = \mcI^{j(t)}\setminus \bar\mcI$
\State $t \gets t+1$
\EndWhile
\end{algorithmic}
\end{algorithm}
\end{center}

These files are the associated experiments for the following Overleaf project:

Progressive Allocation as a Mean-Reverting Process in PSP Auctions (https://www.overleaf.com/read/zvpphnvmqkvq#30ef1a):
