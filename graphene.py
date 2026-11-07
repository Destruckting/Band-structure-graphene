"""
tight binding method for graphene, following the explanation in TB_Saitobook.pdf
Dylan Assinner (dylan.assinner@student.uantwerpen.be) and Ella Bernaerts (ella.bernaerts@student.uantwerpen.be)
last updated: 2026-11-07
"""

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from termcolor import colored

plt.rcParams['text.usetex'] = True

