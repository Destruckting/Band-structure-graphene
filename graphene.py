"""
tight binding method for graphene, following the explanation in TB_Saitobook.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be) and Ella Bernaerts (ella.bernaerts@student.uantwerpen.be)
last updated: 2026-11-11
"""

#from collections import namedtuple
from math import sqrt
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
#from matplotlib.ticker import MultipleLocator
from termcolor import colored

plt.style.use('_mpl-gallery')
plt.rcParams['text.usetex'] = True

class Material:
    def __init__(self, energy2p, t, s):
        self.ep = energy2p
        self.s = s
        self.t = t

    @property
    def ep(self):
        return self._ep

    @ep.setter
    def ep(self, newep):
        if not isinstance(newep, (float, int)):
            raise TypeError(colored('energy 2p must be a number', 'red'))
        self._ep=newep

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, news):
        if not isinstance(news, (float, int)):
            raise TypeError(colored('s must be a number', 'red'))
        self._s=news

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, newt):
        if not isinstance(newt, (float, int)):
            raise TypeError(colored('t must be a number', 'red'))
        self._t=newt

    def Hamiltonian(self, kxa, kya):
        H = np.zeros([2,2], dtype=complex)
        H[0,0] = 1
        H[0,1] = 1
        return H
    def overlap_matrix(self, kxa, kya):
        S = np.zeros([2,2], dtype=complex)
        S[0,0] = 1
        S[1,1] = 1
        return S

class GraphenePz(Material):
    @staticmethod
    def function(kxa, kya):
        return np.exp(1j*kxa/sqrt(3))+2*np.exp(1j*-1*kxa/(2*sqrt(3)))*np.cos(kya/2)

    @staticmethod
    def function_compconjugate(kxa, kya):
        return np.exp(1j*-1*kxa/sqrt(3))+2*np.exp(1j*kxa/(2*sqrt(3)))*np.cos(kya/2)

    def Hamiltonian(self, kxa, kya):
        H = np.zeros([2,2], dtype=complex)
        H[0,0] = self.ep
        H[0,1] = self.t*self.function(kxa, kya)
        H[1,0] = self.t*self.function_compconjugate(kxa, kya)
        H[1,1] = self.ep
        return H

    def overlap_matrix(self, kxa, kya):
        S = np.zeros([2,2], dtype=complex)
        S[0,0] = 1
        S[1,1] = 1
        S[0,1] = self.s*self.function(kxa, kya)
        S[1,0] = self.s*self.function_compconjugate(kxa, kya)
        return S


class SolverTwoDimensions:
    def __init__(self, material, amountx, amounty):
        self.material = material
        if not isinstance(amountx, int) & isinstance(amounty, int):
            raise TypeError(colored('amount must be a positive integer', 'red'))
        if not amountx>0:
            raise ValueError(colored('amountx must be an integer greater than 0', 'red'))
        if not amounty>0:
            raise ValueError(colored('amounty must be an integer greater than 0', 'red'))
        self.kxa=np.linspace(-np.pi, np.pi, amountx)
        self.kya=np.linspace(-np.pi, np.pi, amounty)

    @property
    def material(self):
         return self._material

    @material.setter
    def material(self, newmaterial):
        if not isinstance(newmaterial, Material):
            raise TypeError(colored("material must be an object of a subclass of Material",'red'))
        if type(newmaterial) == Material:
            raise TypeError(colored("material must be an object of a subclass of Material", 'red'))
        self._material = newmaterial

    @property
    def kxa(self):
        return self._kxa

    @kxa.setter
    def kxa(self,newkxa):
        self._kxa=newkxa

    @property
    def kya(self):
        return self._kya

    @kya.setter
    def kya(self, newkya):
        self._kya=newkya

    def eigenvalues_solver(self):
        """Tries to solve the eigenvalues based on the hamiltonian and the overlap matrix"""
        dimensions = (self.material.Hamiltonian(0,0)).shape
        eigenvalues=np.zeros((len(self.kxa),len(self.kya),dimensions[1]))
        for ix,_ in enumerate(self.kxa):
            for iy,_ in enumerate(self.kya):
                eigenval= eigh(self.material.Hamiltonian(ix, iy), self.material.overlap_matrix(ix, iy), eigvals_only=True)
                eigenvalues[ix, iy, :]= eigenval
        return eigenvalues

class Visualizer:
    def __init__(self, solver):
        self.solver = solver

    @property
    def solver(self):
        return self._solver
    @solver.setter
    def solver(self, newsolver):
        if not isinstance(newsolver, SolverTwoDimensions):
            raise TypeError(colored('Solver must be an object of the SolverTwoDimensions class','red'))
        self._solver = newsolver

    def plotter(self):
        eigenvalues=self.solver.eigenvalues_solver()
        _, ax=plt.subplots(subplot_kw={"projection": "3d"})
        x, y = np.meshgrid(self.solver.kxa, self.solver.kya)
        zlow=eigenvalues[:,:,0]
        zhigh=eigenvalues[:,:,1]
        ax.plot_surface(x, y, zhigh, vmin=zlow.min() * 2, cmap="Blues") #looks like shit, check solver?
        ax.set_aspect('equal')




if __name__ == "__main__":
    test=GraphenePz(0, -3.033,0.129)
    testsolve=SolverTwoDimensions(test, 100, 100)
    testplot=Visualizer(testsolve)
    print(testsolve.eigenvalues_solver())
    testplot.plotter()
    plt.show()