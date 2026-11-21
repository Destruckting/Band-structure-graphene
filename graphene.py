"""Tight binding method for graphene.

Following the explanation in TB_Saitobook.pdf, we plot the band structure of
graphene using only the 2p_z orbitals and using the 2s and all 2p orbitals.

Dylan Assinner (dylan.assinner@student.uantwerpen.be) and
Ella Bernaerts (ella.bernaerts@student.uantwerpen.be)

last updated: 2026-11-19
"""

from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from termcolor import colored
from matplotlib.ticker import MultipleLocator

plt.rcParams['text.usetex'] = True


class MaterialTwoDim(ABC):
    """Parent class for 2D materials."""

    def __init__(self, *, energy2p, t, s, amount):
        """Initialize MaterialTwoDim parent class."""
        self.ep = energy2p
        self.s = s
        self.t = t
        self.amount = amount

    @property
    def ep(self):
        """Energy_2p property."""
        return self._ep

    @ep.setter
    def ep(self, newep):
        """Energy_2p setter."""
        if not isinstance(newep, (float, int)):
            raise TypeError(colored('energy 2p must be a number', 'red'))
        self._ep = newep

    @property
    def s(self):
        """S_pi=s property."""
        return self._s

    @s.setter
    def s(self, news):
        """S_pi setter."""
        if not isinstance(news, (float, int)):
            raise TypeError(colored('s must be a number', 'red'))
        self._s = news

    @property
    def t(self):
        """H_pi=t property."""
        return self._t

    @t.setter
    def t(self, newt):
        """H_pi setter."""
        if not isinstance(newt, (float, int)):
            raise TypeError(colored('t must be a number', 'red'))
        self._t = newt

    @property
    def amount(self):
        """Amount property."""
        return self._amount

    @amount.setter
    def amount(self, newamount):
        """Amount setter."""
        if not isinstance(newamount, int):
            raise TypeError(colored('amount must be a '
                                    'positive integer', 'red'))
        if not newamount > 0:
            raise ValueError(colored('amount must be an '
                                     'integer greater than 0', 'red'))
        self._amount = newamount

    @abstractmethod
    def hamiltonian(self, kxa, kya):
        """Hamiltonian matrix."""
        H = np.zeros([2, 2], dtype=complex)
        H[0, 0] = 1
        H[1, 1] = 1
        return H

    @abstractmethod
    def overlap_matrix(self, kxa, kya):
        """Overlap matrix."""
        S = np.zeros([2, 2], dtype=complex)

        S[0, 0] = 1
        S[1, 1] = 1
        return S

    @abstractmethod
    def brilluin_zone(self):
        """Construct the values of the Brilluin zone as two vectors."""
        kxa = np.linspace(-1, 1, self.amount)
        kya = np.linspace(-1, 1, self.amount)
        bz = namedtuple("brilluin_zone", "kxa, kya")
        brilzone = bz(kxa, kya)
        return brilzone

    @abstractmethod
    def high_symmetry_points(self):
        """Save the high symmetry points in a named tuple."""
        highsimpoint = namedtuple('highsimmetrypoints', 'x, y')
        gamma = highsimpoint(0, 0)
        corner = highsimpoint(1, 1)
        mid = highsimpoint(1, 0)
        highsim = namedtuple('highsymmetrypositions', 'gamma, K, M')
        return highsim(gamma, corner, mid)

    def high_symmetry_vector(self):
        """Create a vector that goes from K to gamma to M to K."""
        points = self.high_symmetry_points()
        gam = points.gamma
        korner = points.K
        middle = points.M
        linex = np.concatenate((np.linspace(korner.x, gam.x, 100)[:-1],
                                np.linspace(gam.x, middle.x, 100)[:-1],
                                np.linspace(middle.x, korner.x, 100)))
        liney = np.concatenate((np.linspace(korner.y, gam.y, 100)[:-1],
                                np.linspace(gam.y, middle.y, 100)[:-1],
                                np.linspace(middle.y, korner.y, 100)))
        linevec = namedtuple('High_symmetry_vector', 'x y')
        line = linevec(linex, liney)
        distance_k_to_gamma = np.sqrt((korner.x - gam.x) ** 2 +
                                      (korner.y - gam.y) ** 2)
        distance_gamma_to_m = np.sqrt((gam.x - middle.x) ** 2 +
                                      (gam.y - middle.y) ** 2)
        distance_m_to_k = np.sqrt((middle.x - korner.x) ** 2 +
                                  (middle.y - korner.y) ** 2)
        distancetup = namedtuple('distances', 'long med short')
        distances = distancetup(distance_k_to_gamma,
                                distance_gamma_to_m, distance_m_to_k)
        return line, distances


class GraphenePz(MaterialTwoDim):
    """Class containing the necessary data to solve the band structure
    of graphene using the 2p_z orbitals.
    """

    @staticmethod
    def function(kxa, kya):
        """Make function based on the wave functions of the three
        nearest neighbors following formula e^iK*R.
        """
        return (np.exp(1j * kxa / np.sqrt(3)) + 2 *
                np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))

    def hamiltonian(self, kxa, kya):
        """Hamiltonian matrix."""
        H = np.zeros([2, 2], dtype=complex)
        H[0, 0] = self.ep
        H[0, 1] = self.t * self.function(kxa, kya)
        H[1, 0] = self.t * np.conjugate(self.function(kxa, kya))
        H[1, 1] = self.ep
        return H

    def overlap_matrix(self, kxa, kya):
        """Overlap matrix."""
        S = np.zeros([2, 2], dtype=complex)
        S[0, 0] = 1
        S[1, 1] = 1
        S[0, 1] = self.s * self.function(kxa, kya)
        S[1, 0] = self.s * np.conjugate(self.function(kxa, kya))
        return S

    def brilluin_zone(self):
        """Construct the values of the Brilluin zone as two vectors."""
        kxa = np.linspace(-2*np.pi/np.sqrt(3), 2*np.pi/np.sqrt(3), self.amount)
        kya = np.linspace(-4*np.pi/3, 4*np.pi/3, self.amount)
        bz = namedtuple("brilluin_zone", "kxa, kya")
        brilzone = bz(kxa, kya)
        return brilzone

    def high_symmetry_points(self):
        """Save the high symmetry points in a named tuple."""
        highsimpoint = namedtuple('high_simmetry_coordinates', 'x, y')
        gamma = highsimpoint(0, 0)
        corner = highsimpoint(2 * np.pi / np.sqrt(3), 2 * np.pi / 3)
        mid = highsimpoint(2 * np.pi / np.sqrt(3), 0)
        highsim = namedtuple('high_symmetry_positions', 'gamma, K, M')
        return highsim(gamma, corner, mid)


class Graphene(MaterialTwoDim):
    """Class containing the necessary data to solve the band structure
    of graphene using the 2s and all 2p orbitals.
    """

    def __init__(self, *, energy2s, hss, Sss, hsp, Ssp, hsig, Ssig, **kwargs):
        """Initialize the Graphene class."""
        super().__init__(**kwargs)
        self.hsigma = hsig
        self.hss = hss
        self.hsp = hsp
        self.es = energy2s
        self.sss = Sss
        self.ssp = Ssp
        self.ssigma = Ssig
        # might look into dataclasses to lower amount of instance attributes

    @property
    def hsigma(self):
        """H_sigma property."""
        return self._hsigma

    @hsigma.setter
    def hsigma(self, newhsigma):
        """H_sigma setter."""
        if not isinstance(newhsigma, (float, int)):
            raise TypeError(colored('Hsigma must be a '
                                    'float or integer', 'red'))
        self._hsigma = newhsigma

    @property
    def hss(self):
        """Hss property."""
        return self._hss

    @hss.setter
    def hss(self, newhss):
        """Hss setter."""
        if not isinstance(newhss, (float, int)):
            raise TypeError(colored('Hss must be a '
                                    'float or integer', 'red'))
        self._hss = newhss

    @property
    def hsp(self):
        """Hsp property."""
        return self._hsp

    @hsp.setter
    def hsp(self, newhsp):
        """Hsp setter."""
        if not isinstance(newhsp, (float, int)):
            raise TypeError(colored('Hsp must be a '
                                    'float or integer', 'red'))
        self._hsp = newhsp

    @property
    def es(self):
        """Energy_2s property."""
        return self._es

    @es.setter
    def es(self, newes):
        """Energy_2s setter."""
        if not isinstance(newes, (float, int)):
            raise TypeError(colored('Es must be a '
                                    'float or integer', 'red'))
        self._es = newes

    @property
    def sss(self):
        """Sss property."""
        return self._sss

    @sss.setter
    def sss(self, newsss):
        """Sss setter."""
        if not isinstance(newsss, (float, int)):
            raise TypeError(colored('Sss must be a '
                                    'float or integer', 'red'))
        self._sss = newsss

    @property
    def ssp(self):
        """Ssp property."""
        return self._ssp

    @ssp.setter
    def ssp(self, newssp):
        """Ssp setter."""
        if not isinstance(newssp, (float, int)):
            raise TypeError(colored('Ssp must be a '
                                    'float or integer', 'red'))
        self._ssp = newssp

    @property
    def ssigma(self):
        """Ssigma property."""
        return self._ssigma

    @ssigma.setter
    def ssigma(self, newssigma):
        """Ssigma setter."""
        if not isinstance(newssigma, (float, int)):
            raise TypeError(colored('ssigma must be a '
                                    'float or integer', 'red'))
        self._ssigma = newssigma

    def hamiltonian(self, kxa, kya):
        """Hamiltonian matrix."""
        H = np.zeros([8, 8], dtype=complex)
        H[0, 0] = self.es
        H[1, 1] = self.ep
        H[2, 2] = self.ep
        H[3, 3] = self.ep
        H[4, 4] = self.es
        H[5, 5] = self.ep
        H[6, 6] = self.ep
        H[7, 7] = self.ep
        H[0, 4] = self.hss * (np.exp(1j * kxa / np.sqrt(3)) + 2 *
                              np.exp(-1j * kxa / (2 * np.sqrt(3))) *
                              np.cos(kya / 2))
        H[0, 5] = self.hsp * (-np.exp(1j * kxa / np.sqrt(3)) +
                              np.exp(-1j * kxa / (2 * np.sqrt(3))) *
                              np.cos(kya / 2))
        H[0, 6] = (self.hsp * -1j * np.sqrt(3) *
                   np.exp(-1 * 1j * kxa / (2 * np.sqrt(3))) * np.sin(kya / 2))
        H[1, 4] = -H[0, 5]
        H[1, 5] = (-self.hsigma * np.exp(1j * kxa / np.sqrt(3)) +
                   (-self.hsigma / 2 + 3 * self.t / 2) *
                   np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))
        H[1, 6] = (1j * (np.sqrt(3) / 2) * (self.hsigma + self.t) *
                   np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.sin(kya / 2))
        H[2, 4] = -H[0, 6]
        H[2, 5] = H[1, 6]
        H[2, 6] = (self.t * np.exp(1j * kxa / np.sqrt(3)) +
                   (-3 * self.hsigma / 2 + self.t / 2) *
                   np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))
        H[3, 7] = self.t * (np.exp(1j * kxa / np.sqrt(3)) + 2 *
                            np.exp(-1j * kxa / (2 * np.sqrt(3))) *
                            np.cos(kya / 2))
        for row in range(4, 8):
            for column in range(0, 4):
                H[row, column] = np.conjugate(H[row-4, column+4])
        return H

    def overlap_matrix(self, kxa, kya):
        """Overlap matrix."""
        S = np.zeros([8, 8], dtype=complex)
        for index in range(8):
            S[index, index] = 1
        S[0, 4] = self.sss * (np.exp(1j * kxa / np.sqrt(3)) + 2 *
                              np.exp(-1j * kxa / (2 * np.sqrt(3))) *
                              np.cos(kya / 2))
        S[0, 5] = self.ssp * (-np.exp(1j * kxa / np.sqrt(3)) +
                              np.exp(-1j * kxa / (2 * np.sqrt(3))) *
                              np.cos(kya / 2))
        S[0, 6] = (self.ssp * -1j * np.sqrt(3) *
                   np.exp(-1 * 1j * kxa / (2 * np.sqrt(3))) * np.sin(kya / 2))
        S[1, 4] = -S[0, 5]
        S[1, 5] = (-self.ssigma * np.exp(1j * kxa / np.sqrt(3)) +
                   (-self.ssigma / 2 + 3 * self.s / 2) *
                   np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))
        S[1, 6] = (1j * (np.sqrt(3) / 2) * (self.ssigma + self.s) *
                   np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.sin(kya / 2))
        S[2, 4] = -S[0, 6]
        S[2, 5] = S[1, 6]
        S[2, 6] = (self.s * np.exp(1j * kxa / np.sqrt(3)) +
                   (-3 * self.ssigma / 2 + self.s / 2) *
                   np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))
        S[3, 7] = self.s * (np.exp(1j * kxa / np.sqrt(3)) +
                            2 * np.exp(-1j * kxa / (2 * np.sqrt(3))) *
                            np.cos(kya / 2))
        for row in range(4, 8):
            for column in range(0, 4):
                S[row, column] = np.conjugate(S[row - 4, column + 4])
        return S

    def brilluin_zone(self):
        """Construct the values of the Brilluin zone as two vectors."""
        kxa = np.linspace(-2*np.pi/np.sqrt(3), 2*np.pi/np.sqrt(3), self.amount)
        kya = np.linspace(-4*np.pi/3, 4*np.pi/3, self.amount)
        bz = namedtuple("brilluin_zone", "kxa, kya")
        brilzone = bz(kxa, kya)
        return brilzone

    def high_symmetry_points(self):
        """Save the high symmetry points in a named tuple."""
        highsimpoint = namedtuple('high_simmetry_coordinates', 'x, y')
        gamma = highsimpoint(0, 0)
        corner = highsimpoint(2*np.pi/np.sqrt(3), 2*np.pi/3)
        mid = highsimpoint(2*np.pi/np.sqrt(3), 0)
        highsim = namedtuple('high_symmetry_positions', 'gamma, K, M')
        return highsim(gamma, corner, mid)


class SolverTwoDimensions:
    """Solve the eigenvalues of a two-dimensional material."""

    def __init__(self, material):
        """Initialize SolverTwoDimentions class."""
        self.material = material
        bz = self.material.brilluin_zone()
        self.kxa = bz.kxa
        self.kya = bz.kya
        self.points = material.high_symmetry_points()

    @property
    def material(self):
        """Material property."""
        return self._material

    @material.setter
    def material(self, newmaterial):
        """Setter for material."""
        if not isinstance(newmaterial, MaterialTwoDim):
            raise TypeError(colored("material must be an object of "
                                    "a subclass of MaterialTwoDim", 'red'))
        if newmaterial is MaterialTwoDim:
            # parent class is rejected as this does not give any valuable data
            raise TypeError(colored("material must be an object of "
                                    "a subclass of MaterialTwoDim", 'red'))
        self._material = newmaterial

    def eigenvalues_solver(self):
        """Solve the eigenvalues based on the Hamiltonian and
        the overlap matrix.
        """
        dimensions = (self.material.hamiltonian(0, 0)).shape
        eigenvalues = np.zeros((len(self.kxa), len(self.kya), dimensions[0]))
        for ix, valx in enumerate(self.kxa):
            for iy, valy in enumerate(self.kya):
                eigenval, _ = eigh(self.material.hamiltonian(valx, valy),
                                   self.material.overlap_matrix(valx, valy))
                eigenvalues[ix, iy, :] = eigenval
        return eigenvalues, dimensions[0]

    def eigenvalues_high_simmetry(self):
        """Solve the eigenvalues for the high symmetry directions."""
        dimensions = (self.material.hamiltonian(0, 0)).shape
        highsimvec, _ = self.material.high_symmetry_vector()
        eigenvalues = np.zeros((len(highsimvec.x), dimensions[0]))
        for i, _ in enumerate(highsimvec.x):
            eigenval, _ = eigh(self.material.hamiltonian(highsimvec.x[i],
                                                         highsimvec.y[i]),
                               self.material.overlap_matrix(highsimvec.x[i],
                                                            highsimvec.y[i]))
            eigenvalues[i, :] = eigenval
        return eigenvalues


class Visualizer:
    """Plot the calculated values of the Brilluin zone."""

    def __init__(self, solver):
        """Initialize Visualizer class."""
        self.solver = solver

    @property
    def solver(self):
        """Solver property."""
        return self._solver

    @solver.setter
    def solver(self, newsolver):
        """Setter function for solver."""
        if not isinstance(newsolver, SolverTwoDimensions):
            raise TypeError(colored('Solver must be an object of the '
                                    'SolverTwoDimensions class', 'red'))
        self._solver = newsolver

    def plotter(self):
        """Plot all calculated values in 3D."""
        eigenvalues, dimensions = self.solver.eigenvalues_solver()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, y = np.meshgrid(self.solver.kxa, self.solver.kya, indexing="ij")
        for i in range(dimensions):
            ax.plot_surface(x, y, eigenvalues[:, :, i], color="silver",
                            edgecolor="black", linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax

    def clipped_plotter(self):
        """Show only the Brilluin zone of graphene, everything outside
        the hexagon is set to NaN.
        """
        eigenvalues, dimensions = self.solver.eigenvalues_solver()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, y = np.meshgrid(self.solver.kxa, self.solver.kya, indexing="ij")
        # creating hexagonal mask
        abs_x = np.abs(x)
        abs_y = np.abs(y)
        radius = np.sqrt(self.solver.points.K.x ** 2 +
                         self.solver.points.K.y ** 2)
        outside_hex = ((abs_x > radius * np.sqrt(3) / 2) |
                       (abs_y + abs_x / np.sqrt(3) > radius))
        # plotting eigenvalues
        for i in range(dimensions):
            z = eigenvalues[:, :, i]
            z[outside_hex] = np.nan
            ax.plot_surface(x, y, z, color="silver", edgecolor="black",
                            linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax

    def plotter_highsympoints(self, ax, linecolor='black', labelcolor='black'):
        """Plot the high symmetry triangle on a 3D plot."""
        # functions used to verify positions of corners and edge
        # (to find K and M point)
        points = self.solver.points
        gam = points.gamma
        korner = points.K
        middle = points.M
        ax.plot(np.linspace(0, 2 * np.pi / np.sqrt(3), 100),
                np.linspace(0, 2 * np.pi / 3, 100), np.zeros(100),
                color=linecolor, zorder=20)
        ax.plot(np.linspace(0, 2 * np.pi / np.sqrt(3), 100),
                np.zeros(100), np.zeros(100), color=linecolor, zorder=20)
        ax.plot(2 * np.pi / np.sqrt(3) * np.ones(100),
                np.linspace(0, 2 * np.pi / 3, 100),
                np.zeros(100), color=linecolor, zorder=20)
        ax.scatter(gam.x, gam.y, 0, color=labelcolor, zorder=20)
        ax.text(gam.x, gam.y, 1, r'$\gamma$', ha='center',
                va='center', fontsize=12, zorder=20, color=labelcolor)
        ax.scatter(middle.x, middle.y, 0, color=labelcolor, zorder=20)
        ax.text(middle.x + 0.2, middle.y, 0, 'M', ha='center',
                va='center', fontsize=12, zorder=20, color=labelcolor)
        ax.scatter(korner.x, korner.y, 0, color=labelcolor)
        ax.text(korner.x + 0.2, korner.y + 0.2, 0, 'K', ha='center',
                va='center', fontsize=12, zorder=20, color=labelcolor)

    def highsym_bs(self):
        """Plot the Brilluin zone for the high symmetry directions."""
        _, distances = self.solver.material.high_symmetry_vector()
        eigenval = self.solver.eigenvalues_high_simmetry()
        x = np.concatenate((np.linspace(-distances.long, 0, 100)[:-1],
                            np.linspace(0, distances.med, 100)[:-1],
                            np.linspace(distances.med, distances.short +
                                        distances.med, 100)))
        yliney = np.linspace(eigenval.min() - 1, eigenval.max() + 1, 100)
        ylinegammax = np.zeros(100)
        ylinemx = np.ones(100) * distances.med
        _, ax = plt.subplots()
        ax.plot(x, eigenval, color='black')
        ax.plot(ylinegammax, yliney, color='black', linewidth=0.6)
        ax.plot(ylinemx, yliney, color='black', linewidth=0.6)
        ax.set_ylim((eigenval.min() - 1, eigenval.max() + 1))
        ax.set_xlim((-distances.long, distances.short + distances.med))
        ax.set_xticks((-distances.long, 0, distances.med,
                       distances.med + distances.short),
                      ('K', r'$\Gamma$', 'M', 'K'))
        ax.tick_params(axis='y', which='both', direction='in')
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylabel('E [eV]')
        secondax = ax.secondary_yaxis(location='right')
        secondax.yaxis.set_major_locator(MultipleLocator(5))
        secondax.yaxis.set_minor_locator(MultipleLocator(1))
        secondax.tick_params(axis='y', which='both', direction='in')
        secondax.set_yticklabels([])
        return ax


if __name__ == "__main__":
    # testing solver and plotter using simplified system
    test = GraphenePz(energy2p=0, t=-3.033, s=0.129, amount=500)
    testsolve = SolverTwoDimensions(test)
    testplot = Visualizer(testsolve)
    axtest = testplot.highsym_bs()
    axtest.set_title(r'Brilluin zone Graphene (2p$_z$)'
                     r' for high symmetry points', fontsize=16)
    axestest = testplot.clipped_plotter()
    axestest.set_title(r'Brilluin zone Graphene (2p$_z$)'
                       r' for high symmetry points', fontsize=16)
    # testplot.plotter_highsympoints(axestest, linecolor='maroon',
    #                                labelcolor='red')
    plt.show()
    # actual calculations for full system
    graphene = Graphene(energy2s=-8.868, energy2p=0, hss=-6.769, Sss=0.212,
                        hsp=-5.580, Ssp=0.102, hsig=-5.037, Ssig=0.146,
                        s=0.129, t=-3.033, amount=500)
    graphene_solver = SolverTwoDimensions(graphene)
    graphene_plotter = Visualizer(graphene_solver)
    axes = graphene_plotter.highsym_bs()
    axes.set_title(r'Brilluin zone Graphene (2s, 2p$_x$, 2p$_y$, 2p$_z$)'
                   r' for high symmetry points', fontsize=16)
    axes3D = graphene_plotter.clipped_plotter()
    axes3D.set_title(r'Brilluin zone Graphene (2s, 2p$_x$, 2p$_y$, 2p$_z$)'
                     r' for high symmetry points', fontsize=16)
    # graphene_plotter.plotter_highsympoints(axes3D, linecolor='maroon',
    #                                        labelcolor='red')
    plt.show()
