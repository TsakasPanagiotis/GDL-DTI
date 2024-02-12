# conversion of simple exponential decay to a form that is linear wrt. the d-tensor

$$
\begin{align*}

& S = S_0 \exp(- \ b \ g \ D \ g^T) \\

& \ln ( S / S_0 ) = - \ b \ g \ D \ g^T \\

& \ln ( S / S_0 ) = - \ b \ 
\begin{bmatrix}
g_0 & g_1 & g_2
\end{bmatrix} \ 
\begin{bmatrix}
D_{0,0} & D_{0,1} & D_{0,2} \\
D_{0,1} & D_{1,1} & D_{1,2} \\
D_{0,2} & D_{1,2} & D_{2,2} \\
\end{bmatrix} \ 
\begin{bmatrix}
g_0 \\ g_1 \\ g_2
\end{bmatrix} \\

& \ln ( S / S_0 ) = - \ b \ 
\begin{bmatrix}
g_0 & g_1 & g_2
\end{bmatrix} \
\begin{bmatrix}
g_0 D_{0,0} + g_1 D_{0,1} + g_2 D_{0,2} \\
g_0 D_{0,1} + g_1 D_{1,1} + g_2 D_{1,2} \\
g_0 D_{0,2} + g_1 D_{1,2} + g_2 D_{2,2} \\
\end{bmatrix} \\

& \ln ( S / S_0 ) = - \ b \ ( \ 
g_0 ( g_0 D_{0,0} + g_1 D_{0,1} + g_2 D_{0,2} ) +
g_1 ( g_0 D_{0,1} + g_1 D_{1,1} + g_2 D_{1,2} ) +
g_2 ( g_0 D_{0,2} + g_1 D_{1,2} + g_2 D_{2,2} )
\ ) \\

& \ln ( S / S_0 ) = - \ b \ ( \ 
g_0 g_0 D_{0,0} + g_0 g_1 D_{0,1} + g_0 g_2 D_{0,2} +
g_1 g_0 D_{0,1} + g_1 g_1 D_{1,1} + g_1 g_2 D_{1,2} +
g_2 g_0 D_{0,2} + g_2 g_1 D_{1,2} + g_2 g_2 D_{2,2}
\ ) \\

& \ln ( S / S_0 ) = - \ b \ ( \ 
g_0 g_0 D_{0,0} + 2 g_0 g_1 D_{0,1} + 2 g_0 g_2 D_{0,2} +
g_1 g_1 D_{1,1} + 2 g_1 g_2 D_{1,2} + g_2 g_2 D_{2,2}
\ ) \\

& \ln ( S / S_0 ) = 
\begin{bmatrix}
D_{0,0} & D_{0,1} & D_{0,2} & D_{1,1} & D_{1,2} & D_{2,2} \\
\end{bmatrix} \
(-b) \
\begin{bmatrix}
g_0 g_0 \\ 2 g_0 g_1 \\ 2 b g_0 g_2 \\
g_1 g_1 \\ 2 g_1 g_2 \\ g_2 g_2 \\
\end{bmatrix}
\\

\end{align*}
$$