import numpy
import theano
import theano.tensor as T
from theano import pp
x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
print(pp(gy))

f = theano.function([x], gy)
print(f(4))

print(pp(f.maker.fgraph.outputs[0]))

x = T.dmatrix('x')
s = T.sum(1 / (1 + T.exp(-x)))
gs = T.grad(s, x)
dlogistic = theano.function([x], gs)
print(dlogistic([[0, 1], [-1, -2]]))


t = T.dscalar('t')
w = T.sum(1 / (1 + T.exp(-t)))
gt = T.grad(w, t)
tlogistic = theano.function([t], gt)
print(tlogistic(0))