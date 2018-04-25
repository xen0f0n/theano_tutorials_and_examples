from theano import shared, function
import theano.tensor as T

state = shared(0)
inc = T.iscalar('inc')
# accumulator = function([inc], state, updates=[(state, state + inc)])
accumulator = function([inc], state, updates={state: state+inc})
inc2 = T.iscalar('inc2')
decrementor = function([inc2], state, updates=[(state, state-inc2)])

print("device=", T.config.device)
print("State", state.get_value())
accumulator(1)
print("State", state.get_value())

decrementor(1)
print("State", state.get_value())

fn_of_state = state * 2 + inc
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print("Skip shared", skip_shared(1,3))

print("State", state.get_value())