"""
tight binding method for graphene, following the explanation in TB_Saitobook.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be) and Ella Bernaerts (ella.bernaerts@student.uantwerpen.be)
last updated: 2026-11-11
"""

from collections import namedtuple
from math import sqrt
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from termcolor import colored

plt.style.use('_mpl-gallery')
plt.rcParams['text.usetex'] = True

class MaterialTwoDim:
    def __init__(self, energy2p, t, s, amount):
        self.ep = energy2p
        self.s = s
        self.t = t
        self.amount = amount


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

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, newamount):
        if not isinstance(newamount, int) :
            raise TypeError(colored('amount must be a positive integer', 'red'))
        if not newamount>0:
            raise ValueError(colored('amount must be an integer greater than 0', 'red'))
        self._amount = newamount

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
    def brilluin_zone(self):
        kxa = np.linspace(-1,1,self.amount)
        kya = np.linspace(-1, 1, self.amount)
        bz=namedtuple("brilluin_zone", "kxa, kya")
        brilzone=bz(kxa, kya)
        return brilzone


class GraphenePz(MaterialTwoDim):
    @staticmethod
    def function(kxa, kya):
        return np.exp(1j*kxa/sqrt(3))+2*np.exp(1j*-1*kxa/(2*sqrt(3)))*np.cos(kya/2)

    def Hamiltonian(self, kxa, kya):
        H = np.zeros([2,2], dtype=complex)
        H[0,0] = self.ep
        H[0,1] = self.t*self.function(kxa, kya)
        H[1,0] = self.t*np.conjugate(self.function(kxa, kya))
        H[1,1] = self.ep
        return H

    def overlap_matrix(self, kxa, kya):
        S = np.zeros([2,2], dtype=complex)
        S[0,0] = 1
        S[1,1] = 1
        S[0,1] = self.s*self.function(kxa, kya)
        S[1,0] = self.s*np.conjugate(self.function(kxa, kya))
        return S

    def brilluin_zone(self):
        kxa = np.linspace(-4*np.pi/np.sqrt(3), 4*np.pi/np.sqrt(3), self.amount)
        kya = np.linspace(-4*np.pi/np.sqrt(3), 4*np.pi/np.sqrt(3), self.amount)
        bz = namedtuple("brilluin_zone", "kxa, kya")
        brilzone = bz(kxa, kya)
        return brilzone

class SolverTwoDimensions:
    def __init__(self, material):
        self.material = material
        bz= self.material.brilluin_zone()
        self.kxa = bz.kxa
        self.kya = bz.kya

    @property
    def material(self):
         return self._material

    @material.setter
    def material(self, newmaterial):
        if not isinstance(newmaterial, MaterialTwoDim):
            raise TypeError(colored("material must be an object of a subclass of Material",'red'))
        if type(newmaterial) == MaterialTwoDim:
            raise TypeError(colored("material must be an object of a subclass of Material", 'red'))
        self._material = newmaterial

    def eigenvalues_solver(self):
        """Tries to solve the eigenvalues based on the Hamiltonian and the overlap matrix"""
        dimensions = (self.material.Hamiltonian(0,0)).shape
        eigenvalues=np.zeros((len(self.kxa),len(self.kya),dimensions[1]))
        for ix, valx in enumerate(self.kxa):
            for iy, valy in enumerate(self.kya):
                eigenval, _ = eigh(self.material.Hamiltonian(valx, valy), self.material.overlap_matrix(valx, valy))
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(self.solver.kxa, self.solver.kya, indexing="ij")
        zlow=eigenvalues[:,:,0]
        zhigh=eigenvalues[:,:,1]
        ax.plot_surface(x, y, zhigh, cmap="viridis")
        ax.plot_surface(x, y, zlow, cmap="cividis")



if __name__ == "__main__":
    test=GraphenePz(0, -3.033,0.129, 500)
    testsolve=SolverTwoDimensions(test)
    testplot=Visualizer(testsolve)
    testplot.plotter()
    plt.show()