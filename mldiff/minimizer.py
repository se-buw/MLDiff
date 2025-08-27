from typing import Any, Callable
from z3 import *

PASS = 'GoalNotEntailedByInput'
FAIL = 'GoalEntailedByInput'
UNRESOLVED = 'UNRESOLVED'

def minimizeCore(s : Solver, elements: list[ExprRef], goal : ExprRef) -> list[ExprRef]:
    """Use unsat core to minimize the elements to the solver s, while they guarantee that goal holds."""
    # map to store tracker and element
    trackerToElement = {}
    s.push()
    s.set(unsat_core=True)
    # partial minimization does not do anything here (in particular very bad for logistic regression)
    # s.set("core.minimize_partial", True)
    s.set("core.minimize", True)
    s.add(Not(goal))
    for i in range(len(elements)):
        tracker = Bool('valOfx' + str(i))
        trackerToElement[tracker] = elements[i]
        s.assert_and_track(elements[i], tracker)

    if s.check() == unsat : 
        c = s.unsat_core()
        # map the tracker to the element
        min = []
        for e in c:
            min.append(trackerToElement[e])
    else:
        min = elements
    s.pop()
    return min

def minimize(s : Solver, elements: list[ExprRef], goal : ExprRef) -> list[ExprRef]:
    """Minimize the elements to the solver s, while they guarantee that goal holds."""
    s.push()
    s.add(Not(goal))
    

    def test(inp: list) -> str:
        s.push()
        for e in inp:
            s.add(e)
        res = s.check()
        s.pop()
        if res == sat:
            return PASS
        else:
            return FAIL
    
    min = ddmin(test, elements)  
    s.pop()
    return min

def ddmin(test: Callable, inp: list, *test_args: Any) -> list:
    """Reduce the input inp, using the outcome of test(fun, inp)."""
    assert test(inp, *test_args) != PASS

    n = 2     # Initial granularity
    while len(inp) >= 2:
        start = 0
        subset_length = int(len(inp) / n)
        some_complement_is_failing = False

        while start < len(inp):
            complement = (inp[:int(start)] + inp[int(start + subset_length):])

            if test(complement, *test_args) == FAIL:
                inp = complement
                n = max(n - 1, 2)
                some_complement_is_failing = True
                break

            start += subset_length

        if not some_complement_is_failing:
            if n == len(inp):
                break
            n = min(n * 2, len(inp))

    return inp
