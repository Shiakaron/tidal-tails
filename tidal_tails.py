import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# global variables
Ma = 1
Mb = 1

def compute_energy_of_twobodysystem(y):
    """
    computes energy
    """
    # positions
    ra = np.array([y[0:3]])
    rb = np.array([y[3:6]])
    # velocities
    rdota = np.array([y[6:9]])
    rdotb = np.array([y[9:12]])

    # potenial energy
    r_ab = rb - ra # separation vector
    deltar = np.linalg.norm(r_ab) # separation vector modulus
    PE = - Ma * Mb / deltar

    # kinetic energy
    KE = (Ma / 2) * (np.linalg.norm(rdota) ** 2) + (Mb / 2) * (np.linalg.norm(rdotb) ** 2)

    E = PE + KE
    print("Energies: ", "{:.16f}".format(PE), "{:.16f}".format(KE), "{:.16f}".format(E))
    return E

def two_body_problem_derivatives(t, y):
    """
    function evaluates derivatives for the two body mass problem

    t: time parameter
    y: list of [positions] and [velocities]

    return: [rdots], [rdotdots]
    """
    E = compute_energy_of_twobodysystem(y)
    # positions
    ra = np.array([y[0:3]])
    rb = np.array([y[3:6]])
    # separation
    r_ab = rb - ra
    deltar = np.linalg.norm(r_ab)
    # accelerations
    rdotdota = Mb * r_ab / (deltar ** 3)
    rdotdotb = - Ma * r_ab / (deltar ** 3)
    # rewrite in column form
    ret = np.concatenate((np.array(y[6:]),rdotdota,rdotdotb), axis=None)
    return ret

def ode_solver(fun, t_span, y0, t_eval):
    """
    scipy.integrate.solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, args=None, **options)[source]

    t_span=(t0,tf)

    return: solution of ode solver
    """
    sol = integrate.solve_ivp(fun = fun, t_span = t_span, y0 = y0, t_eval=t_eval)
    return sol

def plot(sol):
    """

    """
    # positions
    xa = sol.y[0]
    ya = sol.y[1]
    xb = sol.y[3]
    yb = sol.y[4]
    plt.plot(xa,ya,xb,yb)
    plt.show()

def main():
    x = 10
    y = 10
    takis_factor = 1

    # circular orbit
    vax = 0
    vay = np.sqrt(Mb*Mb / (2*x*(Ma+Mb))) * takis_factor

    # parabolic orbit from closest approach
    # vax = 0
    # vay = np.sqrt(Mb*Mb / (x*(Ma+Mb))) * takis_factor

    # parabolic orbit from far away, trying to get closes approach to be some desired value
    closestapproach = 10
    vax = np.sqrt(Ma*closestapproach/4) / y
    vay = 0

    # start ode solver
    tf = 10000
    teval = np.arange(0,tf,0.01)
    y0 = np.array([-x,-y,0, x,y,0, vax,vay,0, -vax,-vay,0])
    sol = ode_solver(two_body_problem_derivatives, (0,tf), y0, teval)
    plot(sol)

if (__name__ == '__main__'):
    main()