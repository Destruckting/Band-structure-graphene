"""
tight binding method for graphene, following the explanation in TB_Saitobook.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be) and Ella Bernaerts (ella.bernaerts@student.uantwerpen.be)
last updated: 2026-11-15
"""

from collections import namedtuple
from math import sqrt
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from termcolor import colored

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

    def high_symmetry_points(self):
        highsimpoint= namedtuple('highsimmetrypoints', 'x, y')
        gamma = highsimpoint(0, 0)
        K = highsimpoint(1,1)
        M= highsimpoint(1, 0)
        highsim = namedtuple('highsymmetrypositions', 'gamma, K, M')
        return highsim(gamma, K, M)

    def high_symmetry_vector(self):
        """creates a vector that goes from K to gamma to M to K"""
        points=self.high_symmetry_points()
        gam = points.gamma
        korner = points.K
        middle = points.M
        linex=np.concatenate((np.linspace(korner.x, gam.x, 100)[:-1],
                              np.linspace(gam.x,middle.x,100)[:-1],
                              np.linspace(middle.x, korner.x, 100)))
        liney=np.concatenate((np.linspace(korner.y, gam.y, 100)[:-1],
                              np.linspace(gam.y,middle.y,100)[:-1],
                              np.linspace(middle.y, korner.y, 100)))
        linevec = namedtuple('High_symmetry_vector', 'x y')
        line = linevec(linex, liney)
        distanceKtoGamma=sqrt((korner.x-gam.x)**2+(korner.y-gam.y)**2)
        distanceGammatoM=sqrt((gam.x-middle.x)**2+(gam.y-middle.y)**2)
        distanceMtoK=sqrt((middle.x-korner.x)**2+(middle.y-korner.y)**2)
        distancetup=namedtuple('distances','long med short')
        distances=distancetup(distanceKtoGamma, distanceGammatoM, distanceMtoK)
        return line, distances


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
        kxa = np.linspace(-2*np.pi/np.sqrt(3), 2*np.pi/np.sqrt(3), self.amount)
        kya = np.linspace(-4*np.pi/3, 4*np.pi/3, self.amount)
        bz = namedtuple("brilluin_zone", "kxa, kya")
        brilzone = bz(kxa, kya)
        return brilzone

    def high_symmetry_points(self):
        highsimpoint= namedtuple('highsimmetrypoints', 'x, y')
        gamma = highsimpoint(0, 0)
        K = highsimpoint(2*np.pi/sqrt(3), 2*np.pi/3)
        M= highsimpoint(2*np.pi/sqrt(3), 0)
        highsim = namedtuple('highsymmetrypositions', 'gamma, K, M')
        return highsim(gamma, K, M)


class SolverTwoDimensions:
    def __init__(self, material):
        self.material = material
        bz= self.material.brilluin_zone()
        self.kxa = bz.kxa
        self.kya = bz.kya
        self.points = material.high_symmetry_points()

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
        eigenvalues=np.zeros((len(self.kxa),len(self.kya),dimensions[0]))
        for ix, valx in enumerate(self.kxa):
            for iy, valy in enumerate(self.kya):
                eigenval, _ = eigh(self.material.Hamiltonian(valx, valy), self.material.overlap_matrix(valx, valy))
                eigenvalues[ix, iy, :]= eigenval
        return eigenvalues, dimensions[1]

    def eigenvalues_high_simmetry(self):
        """Tries to solve the eigenvalues for the high simmetry directions"""
        dimensions = (self.material.Hamiltonian(0, 0)).shape
        highsimvec, _ = self.material.high_symmetry_vector()
        eigenvalues=np.zeros((len(highsimvec.x),dimensions[0]))
        for i in range(len(highsimvec.x)):
            eigenval, _=eigh(self.material.Hamiltonian(highsimvec.x[i], highsimvec.y[i]),
                             self.material.overlap_matrix(highsimvec.x[i], highsimvec.y[i]))
            eigenvalues[i,:]=eigenval
        return eigenvalues

    def fermi_energy(self): #might be a nice idea to add as an optional calculation
        pass


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
        eigenvalues, dimensions =self.solver.eigenvalues_solver()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, y = np.meshgrid(self.solver.kxa, self.solver.kya, indexing="ij")
        for i in range(dimensions):
            ax.plot_surface(x, y, eigenvalues[:, :, i], color = "silver", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax

    def clipped_plotter(self):
        eigenvalues, dimensions = self.solver.eigenvalues_solver()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x, y = np.meshgrid(self.solver.kxa, self.solver.kya, indexing="ij")
        #creating hexagonal mask
        abs_x=np.abs(x)
        abs_y=np.abs(y)
        radius = np.sqrt(self.solver.points.K.x ** 2 + self.solver.points.K.y ** 2)
        outside_hex=(abs_x > radius * np.sqrt(3) / 2) | (abs_y + abs_x / np.sqrt(3) > radius)
        for i in range(dimensions):
            z=eigenvalues[:, :, i]
            z[outside_hex] = np.nan
            ax.plot_surface(x, y, z, color="silver", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        return ax

    def plotter_highsimpoints(self, ax):
        # functions used to verify positions of corners and edge (to find K and M point)
        points = self.solver.points
        gam = points.gamma
        korner = points.K
        middle = points.M
        ax.plot(np.linspace(0, 2 * np.pi / np.sqrt(3), 100), np.linspace(0, 2 * np.pi / 3, 100), np.zeros(100),
                color='black', zorder=20)
        ax.plot(np.linspace(0, 2 * np.pi / np.sqrt(3), 100), np.zeros(100), np.zeros(100), color='black', zorder=20)
        ax.plot(2 * np.pi / np.sqrt(3) * np.ones(100), np.linspace(0, 2 * np.pi / 3, 100), np.zeros(100), color='black', zorder=20)
        ax.scatter(gam.x, gam.y, 0, color='black', zorder=20)
        ax.text(gam.x, gam.y, 1, r'$\gamma$', ha='center', va='center', fontsize=12, zorder=20)
        ax.scatter(middle.x, middle.y, 0, color='black', zorder=20)
        ax.text(middle.x + 0.2, middle.y, 0, 'M', ha='center', va='center', fontsize=12, zorder=20)
        ax.scatter(korner.x, korner.y, 0, color='black')
        ax.text(korner.x + 0.2, korner.y + 0.2, 0, 'K', ha='center', va='center', fontsize=12, zorder=20)

    def highsim_bs(self):
        _,distances=self.solver.material.high_symmetry_vector()
        eigenval=self.solver.eigenvalues_high_simmetry()
        x=np.concatenate((np.linspace(-distances.long, 0, 100)[:-1],
                              np.linspace(0, distances.med,100)[:-1],
                              np.linspace(distances.med, distances.short+distances.med, 100)))
        yliney=np.linspace(eigenval.min(),eigenval.max(),100)
        ylinegammax=np.zeros(100)
        ylineMx=np.ones(100)*distances.med
        _,ax=plt.subplots()
        ax.plot(x,eigenval, color='black')
        ax.plot(ylinegammax,yliney, color='black')
        ax.plot(ylineMx,yliney, color='black')
        ax.set_ylabel(r'E [eV]')


if __name__ == "__main__":
    test=GraphenePz(0, -3.033,0.129, 500)
    testsolve=SolverTwoDimensions(test)
    testsolve.eigenvalues_high_simmetry()
    testplot=Visualizer(testsolve)
    testplot.highsim_bs()
    axes = testplot.clipped_plotter()
    testplot.plotter_highsimpoints(axes)
    plt.show()
