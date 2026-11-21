# Graphene Band Structure Using Tight-Binding Models
Simulating the band structure of graphene using tight-binding method

## Project overview
Using the tight binding method, we will calculate the band structure of graphene. 
Using the calculated eigenvalues, we will try to model a DOS function in the Brillouin zone.
Afterward, we will try to use this code to calculate the band structure and DOS for hexagonal Boron Nitride (h-BN)

## Current progress
- [x] tight-binding method in 1D, PolyAcetylene
- [x] Solver and Plotter work on simplified graphene model (only 2Pz orbitals)
- [x] Model created for simulation of graphene (2S, 2Px, 2Py and 2Pz orbitals)
- [ ] DOS calculations based on the calculated eigenvalues
- [ ] expanding the model for h-BN

## Repository structure
| File                        | Description                                                                     |
|-----------------------------|---------------------------------------------------------------------------------|
| 'TB_Saitobook.pdf'          | reference document for tight-binding calculations on polyacetylene and graphene |
| 'example_polyacetylene.py'  | scripts for solving the band structure and DOS for polyacetylene                |
| 'graphene.py'               | scripts for solving the band structure and DOS for graphene                     |
| 'README.md'                 | description of the GitHub repository                                            |

## Requirements
- Python 3.14
- numpy
- SciPy
- matplotlib
- termcolor

## Current results
The eigenvalues of the current graphene model, solver and plotter reproduce the same graphs as seen in reference document 'TB_Saitobook.pdf'

## Authors
Dylan Assinner & Ella Bernaerts
Bachelor fysica, UAntwerpen
