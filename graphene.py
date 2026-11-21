"""
tight binding method for graphene, following the explanation in TB_Saitobook.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be) and Ella Bernaerts (ella.bernaerts@student.uantwerpen.be)
last updated: 2026-11-19
"""

from collections import namedtuple
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from termcolor import colored
from matplotlib.ticker import MultipleLocator

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
        H[1,1] = 1
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
        distanceKtoGamma=np.sqrt((korner.x-gam.x)**2+(korner.y-gam.y)**2)
        distanceGammatoM=np.sqrt((gam.x-middle.x)**2+(gam.y-middle.y)**2)
        distanceMtoK=np.sqrt((middle.x-korner.x)**2+(middle.y-korner.y)**2)
        distancetup=namedtuple('distances','long med short')
        distances=distancetup(distanceKtoGamma, distanceGammatoM, distanceMtoK)
        return line, distances


class GraphenePz(MaterialTwoDim):
    @staticmethod
    def function(kxa, kya):
        return np.exp(1j*kxa/np.sqrt(3))+2*np.exp(-1j*kxa/(2*np.sqrt(3)))*np.cos(kya/2)

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
        K = highsimpoint(2*np.pi/np.sqrt(3), 2*np.pi/3)
        M= highsimpoint(2*np.pi/np.sqrt(3), 0)
        highsim = namedtuple('highsymmetrypositions', 'gamma, K, M')
        return highsim(gamma, K, M)


class Graphene(MaterialTwoDim):
    def __init__(self, energy2s, energy2p, hss, Sss, hsp, Ssp, hsig, Ssig, Spi, Hpi, amount):
        super().__init__(energy2p, Hpi, Spi, amount)
        self.hsigma=hsig
        self.hss=hss
        self.hsp=hsp
        self.es=energy2s
        self.sss=Sss
        self.ssp=Ssp
        self.ssigma=Ssig

    @property
    def hsigma(self):
        return self._hsigma
    @hsigma.setter
    def hsigma(self, newhsigma):
        if not isinstance(newhsigma, (float, int)):
            raise TypeError(colored('Hsigma must be a float or integer','red'))
        self._hsigma=newhsigma

    @property
    def hss(self):
        return self._hss
    @hss.setter
    def hss(self, newhss):
        if not isinstance(newhss, (float, int)):
            raise TypeError(colored('Hss must be a float or integer','red'))
        self._hss=newhss

    @property
    def hsp(self):
        return self._hsp
    @hsp.setter
    def hsp(self, newhsp):
        if not isinstance(newhsp, (float, int)):
            raise TypeError(colored('Hsp must be a float or integer','red'))
        self._hsp=newhsp

    @property
    def es(self):
        return self._es
    @es.setter
    def es(self, newes):
        if not isinstance(newes, (float, int)):
            raise TypeError(colored('Es must be a float or integer','red'))
        self._es=newes

    @property
    def ss(self):
        return self._ss
    @ss.setter
    def ss(self, newss):
        if not isinstance(newss, (float, int)):
            raise TypeError(colored('Ss must be a float or integer','red'))
        self._ss=newss

    @property
    def ssp(self):
        return self._ssp
    @ssp.setter
    def ssp(self, newssp):
        if not isinstance(newssp, (float, int)):
            raise TypeError(colored('Ssp must be a float or integer','red'))
        self._ssp=newssp

    @property
    def ssigma(self):
        return self._ssigma
    @ssigma.setter
    def ssigma(self, newssigma):
        if not isinstance(newssigma, (float, int)):
            raise TypeError(colored('ssigma must be a float or integer','red'))
        self._ssigma=newssigma

    def Hamiltonian(self, kxa, kya):
        H=np.zeros([8,8], dtype=complex)
        H[0,0]=self.es
        H[1,1]=self.ep
        H[2,2]=self.ep
        H[3,3]=self.ep
        H[4,4]=self.es
        H[5,5]=self.ep
        H[6,6]=self.ep
        H[7,7]=self.ep
        H[0,4]=self.hss*(np.exp(1j*kxa/np.sqrt(3))+2*np.exp(-1j*kxa/(2*np.sqrt(3)))*np.cos(kya/2))
        H[0,5]=self.hsp*(-np.exp(1j*kxa/np.sqrt(3))+np.exp(-1j*kxa/(2*np.sqrt(3)))*np.cos(kya/2))
        H[0,6]=self.hsp*-1j*np.sqrt(3)*np.exp(-1*1j*kxa/(2*np.sqrt(3)))*np.sin(kya/2)
        H[1,4]=-H[0,5]
        H[1,5]=-self.hsigma*np.exp(1j*kxa/np.sqrt(3))+(-self.hsigma/2+3*self.t/2)*np.exp(-1j*kxa/(2*np.sqrt(3)))*np.cos(kya/2)
        H[1,6]=1j*(np.sqrt(3)/2)*(self.hsigma+self.t)*np.exp(-1j*kxa/(2*np.sqrt(3)))*np.sin(kya/2)
        H[2,4]=-H[0,6]
        H[2,5]=H[1,6]
        H[2,6]=self.t*np.exp(1j*kxa/np.sqrt(3))+(-3*self.hsigma/2+self.t/2)*np.exp(-1j*kxa/(2*np.sqrt(3)))*np.cos(kya/2)
        H[3,7]=self.t*(np.exp(1j*kxa/np.sqrt(3))+2*np.exp(-1j*kxa/(2*np.sqrt(3)))*np.cos(kya/2))
        for row in range(4,8):
            for column in range(0,4):
                H[row,column]=np.conjugate(H[row-4, column+4])
        return H

    def overlap_matrix(self, kxa, kya):
        S = np.zeros([8, 8], dtype=complex)
        for index in range(8):
            S[index,index]=1
        S[0, 4] = self.sss * (
                    np.exp(1j * kxa / np.sqrt(3)) + 2 * np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))
        S[0, 5] = self.ssp * (-np.exp(1j * kxa / np.sqrt(3)) + np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))
        S[0, 6] = self.ssp * -1j * np.sqrt(3) * np.exp(-1 * 1j * kxa / (2 * np.sqrt(3))) * np.sin(kya / 2)
        S[1, 4] = -S[0, 5]
        S[1, 5] = -self.ssigma * np.exp(1j * kxa / np.sqrt(3)) + (-self.ssigma / 2 + 3 * self.s / 2) * np.exp(
            -1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2)
        S[1, 6] = 1j * (np.sqrt(3) / 2) * (self.ssigma + self.s) * np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.sin(
            kya / 2)
        S[2, 4] = -S[0, 6]
        S[2, 5] = S[1, 6]
        S[2, 6] = self.s * np.exp(1j * kxa / np.sqrt(3)) + (-3 * self.ssigma / 2 + self.s / 2) * np.exp(
            -1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2)
        S[3, 7] = self.s * (np.exp(1j * kxa / np.sqrt(3)) + 2 * np.exp(-1j * kxa / (2 * np.sqrt(3))) * np.cos(kya / 2))
        for row in range(4, 8):
            for column in range(0, 4):
                S[row, column] = np.conjugate(S[row - 4, column + 4])
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
        K = highsimpoint(2*np.pi/np.sqrt(3), 2*np.pi/3)
        M= highsimpoint(2*np.pi/np.sqrt(3), 0)
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
        return eigenvalues, dimensions[0]

    def eigenvalues_high_simmetry(self):
        """Tries to solve the eigenvalues for the high symmetry directions"""
        dimensions = (self.material.Hamiltonian(0, 0)).shape
        highsimvec, _ = self.material.high_symmetry_vector()
        eigenvalues=np.zeros((len(highsimvec.x),dimensions[0]))
        for i in range(len(highsimvec.x)):
            eigenval, _=eigh(self.material.Hamiltonian(highsimvec.x[i], highsimvec.y[i]),
                             self.material.overlap_matrix(highsimvec.x[i], highsimvec.y[i]))
            eigenvalues[i,:]=eigenval
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

    def highsym_bs(self):
        _,distances=self.solver.material.high_symmetry_vector()
        eigenval=self.solver.eigenvalues_high_simmetry()
        x=np.concatenate((np.linspace(-distances.long, 0, 100)[:-1],
                              np.linspace(0, distances.med,100)[:-1],
                              np.linspace(distances.med, distances.short+distances.med, 100)))
        yliney=np.linspace(eigenval.min()-1,eigenval.max()+1,100)
        ylinegammax=np.zeros(100)
        ylineMx=np.ones(100)*distances.med
        _,ax=plt.subplots()
        ax.plot(x,eigenval, color='black')
        ax.plot(ylinegammax,yliney, color='black', linewidth=0.6)
        ax.plot(ylineMx,yliney, color='black', linewidth=0.6)
        ax.set_ylim((eigenval.min()-1, eigenval.max()+1))
        ax.set_xlim((-distances.long, distances.short+distances.med))
        ax.set_xticks((-distances.long,0,distances.med,distances.med+distances.short),('K',r'$\Gamma$', 'M', 'K'))
        ax.tick_params(axis='y', which='both', direction='in')
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylabel('E [eV]')
        secondax=ax.secondary_yaxis(location='right')
        secondax.yaxis.set_major_locator(MultipleLocator(5))
        secondax.yaxis.set_minor_locator(MultipleLocator(1))
        secondax.tick_params(axis='y', which='both', direction='in')
        secondax.set_yticklabels([])
        return ax



if __name__ == "__main__":
    # test=GraphenePz(0, -3.033,0.129, 500)
    # testsolve=SolverTwoDimensions(test)
    # testplot=Visualizer(testsolve)
    # testplot.highsim_bs()
    # axes = testplot.clipped_plotter()
    # # testplot.plotter_highsimpoints(axes)
    # plt.show()
    material = Graphene(energy2s=-8.868,energy2p= 0, hss=-6.769,
                        Sss=0.212, hsp=-5.580, Ssp=0.102,
                        hsig=-5.037, Ssig=0.146, Hpi=-3.033,
                        Spi=0.129, amount=500)
    solver= SolverTwoDimensions(material)
    plotter= Visualizer(solver)
    axes=plotter.highsym_bs()
    axes.set_title(r'Brilluin zone Graphene (2s, 2p$_x$, 2p$_y$, 2p$_z$) for high symmetry points', fontsize=16)
    plt.show()