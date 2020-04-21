import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import os
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
    print("Energies: ", "{:.16f}".format(PE), "{:.16f}".format(KE), "{:.16f}".format(E), deltar)
    return E

def two_body_problem_derivatives(t, y):
    """
    function evaluates derivatives for the two body mass problem

    t: time parameter
    y: list of [positions] and [velocities]

    return: [rdots], [rdotdots]
    """
    # E = compute_energy_of_twobodysystem(y)
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
    sol = integrate.solve_ivp(fun = fun, t_span = t_span, y0 = y0, t_eval=t_eval, method="Radau")
    return sol

def makegif(sol, tag, lim):
    """
    animation of trajectories
    """
    # positions
    xa = sol.y[0]
    ya = sol.y[1]
    xb = sol.y[3]
    yb = sol.y[4]

    #initialise figure
    fig = plt.figure(figsize=(8,8))
    ax1 = plt.axes(xlim=(-lim,lim), ylim=(-lim,lim))
    # line, = ax1.plot([], [])

    plotlays, plotcols = [2], ["black","red"]
    lines = [] # will contain all the line objects
    for index in range(2):
        lobj = ax1.plot([],[],lw=0.5,color=plotcols[index],markersize=10,marker="o")[0]
        lines.append(lobj)
    #lines.append(ax1.plot([],[],linestyle="",color=plotcols[-1],markersize=10,marker="o")[0])

    def init():
        # function used to initialise line_objects to null data (maybe this is useless but keep just in case)
        for line in lines:
            line.set_data([],[])
        return lines

    dataperframe = 200 # to speed up the animation i set it to ignore every 200 data points
    def animate(i):
        # function used to plot each frame
        up_to_point = i*dataperframe
        ax1.set_xlabel(i) # label the frame (can be removed later)
        xlist = [xa[:up_to_point], xb[:up_to_point] ] # contains the new data for the frame
        ylist = [ya[:up_to_point], yb[:up_to_point] ]
        for lnum,line in enumerate(lines):
            line.set_data(xlist[lnum], ylist[lnum]) # set data for each line separately.
            line.set_markevery((up_to_point-1,up_to_point))
        return lines

    anim = FuncAnimation(fig, animate, frames=int(len(xa)/dataperframe), interval = 5, init_func=init, blit=True)
    path = os.getcwd()+"\\twobody_animation_"+tag+"_.gif"
    anim.save(path,writer='imagemagick')

def plot(sol):
    """
    plot of trajectories
    """
    # positions
    xa = sol.y[0]
    ya = sol.y[1]
    xb = sol.y[3]
    yb = sol.y[4]
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(xa,ya,xb,yb)
    plt.show()

def main():
    # circular orbit
    solve_circular_orbit()

    # parabolic orbit from closest approach
    # vax = 0
    # vay = np.sqrt(Mb*Mb / (x*(Ma+Mb)))

    # parabolic orbit from far away, trying to get closes approach to be some desired value
    # x = 40
    # y = 15
    # vax = np.sqrt(Mb*Mb / ((Ma+Mb)*np.sqrt(x*x+y*y)))
    # vay = 0

    # start ode solver
    # tf = 1000
    # teval = np.arange(0,tf,0.01)
    # y0 = np.array([-x,-y,0, x,y,0, vax,vay,0, -vax,-vay,0])
    # sol = ode_solver(two_body_problem_derivatives, (0,tf), y0, teval)
    # plot(sol)
    # makegif(sol,circular,)

def solve_circular_orbit():
    # circular orbit
    x = 10
    y = 0
    vax = 0
    vay = np.sqrt(Mb*Mb / (2*x*(Ma+Mb)))
    # start ode solver
    tf = 1000
    teval = np.arange(0,tf,0.01)
    y0 = np.array([-x,-y,0, x,y,0, vax,vay,0, -vax,-vay,0])
    sol = ode_solver(two_body_problem_derivatives, (0,tf), y0, teval)
    #plot(sol)
    makegif(sol,"circular_orbit",x+2)


if (__name__ == '__main__'):
    main()
