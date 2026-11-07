"""
tight binding method for polyacetylene,
following the explanation in TB_Saitobook.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be)
last updated: 2026-11-07
"""
from collections import namedtuple
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from termcolor import colored

plt.rcParams['text.usetex'] = True


class PolyAcetyleneTightBinding:
    """
    Class for solving the band structure of polyacetylene
    using the tight binding method
    """
    def __init__(self, energy2p, t, s, amountofpoints):
        """initialize class"""
        self.energy2p = energy2p
        self.t = t
        self.s = s
        self.amount = amountofpoints
        self.ka = np.linspace(-np.pi, np.pi, self.amount)
        self.eigenvaluepos = 0
        self.eigenvalueneg = 0

    @property
    def energy2p(self):
        """energy2p property"""
        return self._energy2p

    @energy2p.setter
    def energy2p(self, newenergy2p):
        """energy2p setter"""
        if not isinstance(newenergy2p, (float, int)):
            raise TypeError(colored('energy2p must be an integer'
                                    ' or a float', 'red'))
        self._energy2p = newenergy2p

    @property
    def t(self):
        """t property"""
        return self._t

    @t.setter
    def t(self, newt):
        """t setter"""
        if not isinstance(newt, (float, int)):
            raise TypeError(colored('t must be an integer or a float', 'red'))
        self._t = newt

    @property
    def s(self):
        """s property"""
        return self._s

    @s.setter
    def s(self, news):
        """s setter"""
        if not isinstance(news, (float, int)):
            raise TypeError(colored('s must be an integer or a float', 'red'))
        self._s = news

    def solved_eigenvalues(self):
        """Solving the eigenvalues using the tight binding method"""
        self.eigenvaluepos = ((self.energy2p + 2 * self.t *
                               np.cos(self.ka / 2)) /
                              (1 + 2 * self.s * np.cos(self.ka / 2)))
        self.eigenvalueneg = ((self.energy2p - 2 * self.t *
                               np.cos(self.ka / 2)) /
                              (1 - 2 * self.s * np.cos(self.ka / 2)))
        eigval = namedtuple('eigenvalues', 'pos neg')
        eigenvalues = eigval(self.eigenvaluepos, self.eigenvalueneg)
        return eigenvalues

    def solver_determinant(self):
        """Solving the determinant of the tight binding method
        (solving the eigenvalues of a general eigenvalue problem)"""
        energies=[]
        for k in self.ka:
            A=np.array([[self.energy2p, 2 * self.t * np.cos(k / 2)], [2 * self.t * np.cos(k / 2), self.energy2p]])
            B=np.array([[1, 2 * self.s * np.cos(k / 2)], [2 * self.s * np.cos(k / 2), 1]])
            #solve det(A-Bl)=0 with l the eigenvalues
            eigenvals, _ = eigh(A,B)
            energies.append(eigenvals)
        band_energies = np.array(energies)
        return band_energies

    def density_of_states(self):
        """calculating the density of states using a histogram"""
        hist, bin_edges = np.histogram(self.solver_determinant(),500)
        return hist/self.amount, bin_edges


    def plot_solved(self):
        """plotting the band structure from the expression of the eigenvalues"""
        x = np.linspace(-1, 1, 100)
        y = self.solved_eigenvalues()
        _, ax = plt.subplots()
        ax.plot(x, y.pos, color='xkcd:bright blue', marker='none',
                label='positive eigenvalues')
        ax.plot(x, y.neg, color='xkcd:deep blue', marker='none',
                label='negative eigenvalues')
        ax.set_xlabel(r'$\frac{ka}{\pi}$')
        ax.set_ylabel(r'$E$')
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_xlim(-1, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_ylim(-2, 4)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.margins(0, 0)
        ax.title.set_text('Band structure polyacetylene')
        ax.legend()
        plt.show()

    def plot_detsolver(self):
        """plotting the band structure from the determinant solver method"""
        x = np.linspace(-1, 1, 500)
        y = self.solver_determinant()
        _, ax = plt.subplots()
        ax.plot(x, y, color='black', marker='none',
                label='eigenvalues')
        ax.set_xlabel(r'$\frac{ka}{\pi}$')
        ax.set_ylabel(r'$E$')
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_xlim(-1, 1)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_ylim(-2, 4)
        ax.yaxis.set_major_locator(MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        ax.margins(0, 0)
        ax.title.set_text('Band structure polyacetylene')
        ax.legend()
        plt.show()



if __name__ == "__main__":
    test = PolyAcetyleneTightBinding(0, -1, 0.2, 500)
    #test.plot_solved()
    test.plot_detsolver()
    val, bin = test.density_of_states()
    plt.stairs(val,bin)
    plt.show()