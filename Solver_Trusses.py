#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:34:19 2021

@author: kendrick
"""

import numpy as np
import math
# compute unknown displacements 
def ComputeDisplacements(K, F, n_unknowns):
    # extract submatrix of unknowns
    K11 = K[0:n_unknowns,0:n_unknowns]
    F1 = F[0:n_unknowns]
    
    d = np.linalg.solve(K11,F1)
    
    return d

# postprocess the forces at known displacement nodes
def PostprocessReactions(K, d, F, n_unknowns, nodes):
    # These are computed net forces and do not
    # take into account external loads applied
    # at these nodes
    F = np.matmul(K[n_unknowns:,0:n_unknowns], d)
    
    # Postprocess the reactions
    for node in nodes:
        if node.xidx >= n_unknowns:
            node.AddReactionXForce(F[node.xidx-n_unknowns][0] - node.xforce_external)
        if node.yidx >= n_unknowns:
            node.AddReactionYForce(F[node.yidx-n_unknowns][0] - node.yforce_external)
        
    return F

# determine internal member loads
def ComputeMemberForces(bars):
    for bar in bars:
        E = bar.E
        A = bar.A
        L = bar.Length()
        [lambdax, lambday] = bar.LambdaTerms()
        
        # Extract global displacements for near (init) and far (end) nodes
        u_Nx = bar.init_node.xdisp
        u_Ny = bar.init_node.ydisp
        u_Fx = bar.end_node.xdisp
        u_Fy = bar.end_node.ydisp
        
        # Calculate member force using textbook Equation 14-23 formulation
        # Note: L is in ft, displacements are in ft. 
        # The ratio (displacement / length) is unitless, so E*A (in kips) dictates the output unit (kips).
        force = (E * A / L) * (lambdax * (u_Fx - u_Nx) + lambday * (u_Fy - u_Ny))
        
        # Store the calculated axial load
        bar.axial_load = force 
    return
    
# compute the normal stresses
def ComputeNormalStresses(bars):
    for bar in bars:
        # Stress is force divided by cross-sectional area
        bar.normal_stress = bar.axial_load / bar.A
    return

# compute the critical buckling load of a member
def ComputeBucklingLoad(bars):
    for bar in bars:
        E = bar.E
        # Members will buckle about their weak axis, so select the minimum moment of inertia
        I = bar.GetWeakSecondMoment()
        
        # Ensure units are consistent: E is in ksi, I is in in^4, but L is currently in feet
        L_inches = bar.Length() * 12.0
        K = 1.0 # Effective length factor is 1.0 for pin-ended trusses
        
        # Critical buckling load formula
        P_cr=(math.pi**2 * E * I) / ((K * L_inches)**2)
        
        bar.buckling_load = P_cr
    return
