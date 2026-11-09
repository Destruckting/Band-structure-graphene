"""
tight binding method for graphene, following the explanation in TB_Saitobook.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be) and Ella Bernaerts (ella.bernaerts@student.uantwerpen.be)
last updated: 2026-11-07
"""

from collections import namedtuple
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from termcolor import colored

plt.rcParams['text.usetex'] = True

class Material:
    def __init__(self, energy2p, s, t):
        self.e2p = energy2p
        self.s = s
        self.t = t
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
    def function(self, kxa, kya):
        return np.exp(1j*kxa/sqrt(3))+2*np.exp(1j*-1*kxa/(2*sqrt(3)))*np.cos(kya/2)
    def function_compconjugate(self, kxa, kya):
        return np.exp(1j*-1*kxa/sqrt(3))+2*np.exp(1j*kxa/(2*sqrt(3)))*np.cos(kya/2)

    def Hamiltonian(self, kxa, kya):
        H = np.zeros([2,2], dtype=complex)
        H[0,0] = self.e2p
        H[0,1] = self.t*self.function(kxa, kya)
        H[1,0] = self.t*self.function_compconjugate(kxa, kya)
        H[1,1] = self.e2p
        return H

    def overlap_matrix(self, kxa, kya):
        S = np.zeros([2,2], dtype=complex)
        S[0,0] = 1
        S[1,1] = 1
        S[0,1] = self.s*self.function(kxa, kya)
        S[1,0] = self.s*self.function_compconjugate(kxa, kya)
        return S


class SolverTwoDimensions:
    def __init__(self, material, amount):
        self.material = material
        self.kxa=np.linspace(-np.pi, np.pi, amount)
        self.kya=np.linspace(-np.pi, np.pi, amount)

    @property
    def material(self):
         return self._material

    @material.setter
    def material(self, newmaterial):
        if not isinstance(newmaterial, Material):
            raise TypeError(colored("new material must be an object of a subclass of Material",'red'))
        if type(newmaterial) == Material:
            raise TypeError(colored("new material must be an object of a subclass of Material", 'red'))
        self._material = newmaterial




if __name__ == "__main__":
    test=GraphenePz(0, -3.033,0.129)
    testing=SolverTwoDimensions(test)
    print(test.Hamiltonian(0,0))