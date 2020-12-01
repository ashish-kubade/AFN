from .SRSolver import SRSolver as SRSolver
from .SRSolver_debug import SRSolver as SRSolver_debug

def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver(opt)
    else:
        raise NotImplementedError

    return solver

def create_solver_db(opt):
    if opt['mode'] == 'sr':
        solver = SRSolver_debug(opt)
    else:
        raise NotImplementedError

    return solver