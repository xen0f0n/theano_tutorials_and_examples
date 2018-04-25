from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

srng = RandomStreams(seed = 234)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)
nearly_zeros = function([], rv_u + rv_u -2 * rv_u)

print(f())
print(f())
print(g())
print(g())
print(".....")
print(nearly_zeros())

srng.seed(902340)
print(f())
print(f())
print(g())
print(g())

state_after_v0 = rv_u.rng.get_value().get_state()
print(nearly_zeros())
v1 = f()
rng = rv_u.rng.get_value(borrow=True)
rng.set_state(state_after_v0)
rv_u.rng.set_value(rng, borrow=True)
print(rng)
v2 = f()             # v2 != v1
v3 = f()             # v3 == v1
print(v2)
print(v3)