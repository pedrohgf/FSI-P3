function G = Multiquadratic(U,V)

A = sum(U .* U, 2);
B = -2 *U * V';
C = sum(V .* V, 2);
G = bsxfun(@plus, A, B);
G = bsxfun(@plus, G, C);
G = 1 + G;
G = sqrt(G);

end