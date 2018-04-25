import numpy
import theano.tensor as T
import pygpu
from theano import function, pp, In
from matplotlib import pyplot

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
t = x * y
f = function([x, In(y, value=1)], [z, t])

print("device=", T.config.device)

print(f(2, 3))
print(numpy.allclose(f(16.3, 12.1), 28.4))
print(x.type)
print(type(x))
print(pp(z))
print(pp(x))

# x = T.dmatrix('x')
# s = 1 / (1 + T.exp(-x))
# logistic = function([x], s)
# print(pp(s))
# x = (logistic([[0, 1], [-1, -2]]))
# print(x)
#
