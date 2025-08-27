from z3 import *


def evaluateDiffMatrix(s: Solver, class1: ArithRef, class2: ArithRef, numClasses: int):
    matrix = [[0 for x in range(numClasses)] for y in range(numClasses)]
    # find all solutions where clf1 and clf2 agree
    s.push()
    while s.check() == sat:
        cls1 = s.model().eval(class1).as_long()
        cls2 = s.model().eval(class2).as_long()
        matrix[cls1][cls2] = 1
        # block the current solution
        s.add(Not(And(class1 == IntVal(cls1), class2 == IntVal(cls2))))
        # print(cls1, cls2)
    s.pop()
    return matrix


def evaluateDiffMatrixNoBlock(
    s: Solver, class1: ArithRef, class2: ArithRef, numClasses: int
):
    matrix = [[0 for x in range(numClasses)] for y in range(numClasses)]
    # find all solutions where clf1 and clf2 agree
    for i in range(numClasses):
        for j in range(numClasses):
            s.push()
            s.add(class1 == i, class2 == j)
            if s.check() == sat:
                matrix[i][j] = 1
            s.pop()
            # print(i, j)
    return matrix


def evaluateDiffMatrixConfirm(
    s: Solver, class1: ArithRef, clf1, class2: ArithRef, clf2, numClasses: int, confirm
):
    matrix = [[0 for x in range(numClasses)] for y in range(numClasses)]
    # find all solutions where clf1 and clf2 agree
    s.push()
    while s.check() == sat:
        cls1 = s.model().eval(class1).as_long()
        cls2 = s.model().eval(class2).as_long()
        if confirm(s, class1, clf1, class2, clf2):
            matrix[cls1][cls2] = 1
        else:
            matrix[cls1][cls2] = -1
        # block the current solution
        s.add(Not(And(class1 == IntVal(cls1), class2 == IntVal(cls2))))
        # print(cls1, cls2)
    s.pop()
    return matrix


def evaluateDiffMatrixIdentify(
    s: Solver,
    class1: ArithRef,
    clf1,
    class2: ArithRef,
    clf2,
    numClasses: int,
    identify,
    block,
):
    matrix = [[0 for x in range(numClasses)] for y in range(numClasses)]
    # find all solutions where clf1 and clf2 agree
    s.push()
    while s.check() == sat:
        cls1, cls2 = identify(s, clf1, clf2)
        # block the actual solution area
        block(s)
        if matrix[cls1][cls2] == 0:
            matrix[cls1][cls2] = 1
            pp(matrix)
        # block the identified solution
        s.add(Not(And(class1 == IntVal(str(cls1)), class2 == IntVal(str(cls2)))))
        # print(cls1, cls2)
    s.pop()
    return matrix
