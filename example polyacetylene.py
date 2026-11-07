"""
tight binding method for polyacetylene, following the explanation in TB_Saitobool.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be)
last updated: 2026-11-06
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

class polyacetylene_TB:
    def __init__(self,energy2p,t,s):
        self.energy2p=energy2p
        self.t=t
        self.s=s
        self.ka=np.linspace(-np.pi,np.pi,100)

    @property
    def energy2p(self):
        return self._energy2p

    @energy2p.setter
    def energy2p(self,newenergy2p):
        if not isinstance(newenergy2p, (float, int)):
            raise TypeError(colored('energy2p must be an integer'
                                    ' or a float', 'red'))
        self._energy2p = newenergy2p

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self,newt):
        if not isinstance(newt, (float,int)):
            raise TypeError(colored('t must be an integer or a float','red'))
        self._t=newt

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self,news):
        if not isinstance(news, (float, int)):
            raise TypeError(colored('s must be an integer or a float','red'))
        self._s=news

    def solver(self):
        self.eigenvaluepos=(self.energy2p+2*self.t*np.cos(self.ka/2))/(1+2*self.s*np.cos(self.ka/2))
        self.eigenvalueneg=(self.energy2p-2*self.t*np.cos(self.ka/2))/(1-2*self.s*np.cos(self.ka/2))
        eigval=namedtuple('eigenvalues','pos neg')
        eigenvalues=eigval(self.eigenvaluepos,self.eigenvalueneg)
        return eigenvalues

    def plot(self):


if __name__=="__main__":
    print('Hello world')
    test=polyacetylene_TB(0,-1,0.2)
    print(test.solver())