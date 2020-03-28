# A file with useful functions for STEVE calculations, primarily the photochemical model

import unicodedata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def reaction_input_output(df):
    '''
    Parse the raw reactions list and return some useful stuff:
    species   - list(str) - all of the species to track
    emissions - list(str) - all of the emissions to track
    din   - dict - a list of reactants for each reaction
    dout  - dict - a list of products for each reaction
    demis - dict - a lits of emissions for each reaction
    '''
    
    cols = []
    for i in df.index:
        s = df.loc[i, 'Reaction']
        inout = s.split(' → ')
        ins = inout[0].split(' + ')
        outs = inout[1].split(' + ')
        ins = [x.strip() for x in ins]
        outs =  [x.strip() for x in outs]
        cols.extend(ins)
        cols.extend(outs)
    cols = list(set(cols)) # unique
    cols.sort()
    # # Remove photons

    species = np.array([s for s in cols if 'hv' not in s])
    emissions = np.array([s for s in cols if 'hv' in s])
    species_ions = np.array([s for s in species if '+' in s])
    species_neutrals = np.array([s for s in species if not '+' in s and s!='e'])

    N = len(species)
    Ne = len(emissions)
    M = len(df)
    dfin  = pd.DataFrame(index=df['Reaction'], columns=species, data=np.zeros((M,N),dtype=bool) )
    dfout = pd.DataFrame(index=df['Reaction'], columns=species, data=np.zeros((M,N),dtype=bool) )
    dfemis= pd.DataFrame(index=df['Reaction'], columns=emissions, data=np.zeros((M,Ne),dtype=bool) )
    for i in df.index:
        s = df.loc[i, 'Reaction']
        inout = s.split(' → ')
        ins = inout[0].split(' + ')
        outs = inout[1].split(' + ')
        ins = [x.strip() for x in ins]
        outs =  [x.strip() for x in outs]
        for x in ins:
            dfin.loc[s, x] = True
        for x in outs:
            if x in species:
                dfout.loc[s, x] = True
            if x in emissions:
                dfemis.loc[s,x] = True
                
    # This code added after I created the above DataFrames, to be converted back to dicts, 
    # because I found that easier. I'm keeping it inelegant like this in case I ever want
    # to use DataFrames again
    # Convert from DataFrame to dict, since that simplifies things.
    # Each (key,value) pair is (reaction, list of relevant reactants or products or emissions)
    din = {}
    dout = {}
    demis = {}
    for rx in dfin.index:
        din[rx]   = dfin.columns[dfin.loc[rx]].values
        dout[rx]  = dfout.columns[dfout.loc[rx]].values
        demis[rx] = dfemis.columns[dfemis.loc[rx]].values
        
        
    # Convert din, dout, demis to species indices instead of species string keys
    reactions = list(din.keys())
    nin = {}
    nout = {}
    nemis = {}
    for rx in reactions:
        xin = []
        xout = []
        xemis = []
        for s in din[rx]:
            j, = np.where(species==s)
            xin.append(j[0])
        for s in dout[rx]:
            j, = np.where(species==s)
            xout.append(j[0])
        for s in demis[rx]:
            j, = np.where(emissions==s)
            xemis.append(j[0])
        nin[rx] = np.array(xin)
        nout[rx] = np.array(xout)
        nemis[rx] = np.array(xemis)
        
    # Dictionary to convert species name to index
    sidx = {}
    for s in species:
        sidx[s] = np.where(species==s)[0][0]
                
    return species, emissions, din, dout, demis, nin, nout, nemis, sidx



def eq_dens(s, n, drc, nin, nout, species):
    '''
    Compute equilibrium concentration for species s using densities and reaction rates given. Intended
    to be used to compute minor species concentrations.
    
    s - str - species to compute (e.g., 'NO', 'O+(2D)')
    n - (Nz,Ns) array - densities
    drc - dictionary of reaction rate coefficients, already evaluated as scalars, not formulas
    species - list of str - the list of species names
    '''
    reactions = list(drc.keys())
    Nz,Ns = np.shape(n)
    js = np.where(species==s)[0][0]

    prodrate = np.zeros(Nz) # cm^3/s
    lossrate = np.zeros(Nz) # 1/s
    for jr, rx in enumerate(reactions): # loop over reactions
        # Find product of input species densities and multiply by reaction coefficient
        if js in nout[rx] and js in nin[rx]: # does not affect equilibrium: exclude
            continue
        if js in nout[rx]: # It's a production rate
            prodrate += drc[rx] * np.array([n[:,jx] for jx in nin[rx]]).prod(axis=0)
        if js in nin[rx]: # It's a loss rate
            # Note: excluding the species under consideration from the loss rate, since it's 
            # the unknown we're solving for
            lossrate += drc[rx] * np.array([n[:,jx] for jx in nin[rx] if jx!=js]).prod(axis=0) 
    neq = prodrate/lossrate
    return neq
    
    
    
def compute_change(n, drc, nin, nout, nemis, Nz, Ns, Nr, Ne):
    '''
    Compute derivative of density and other information. This is what's needed right
    before taking a timestep in the model.
    '''
    reactions = list(drc.keys())
    
    # Compute reaction rate from rate coeff and densities
    # Three variables for diagnostics. Really only dndt is needed for timestepping, I think
    r = np.zeros((Nz, Nr)) # Rate of each reaction cm^3/s
    dndt = np.zeros((Nz, Ns)) # Rate of change for each species
    dndtfull = np.zeros((Nz, Nr, Ns)) # cm^3/s, contribution of each reaction to the rate of each species 
                                      # This might be too slow to use in practice
    ver = np.zeros((Nz,Ne)) # Volume emission rate, resetting to zero each time step
    verfull = np.zeros((Nz,Nr,Ne))
    for jr, rx in enumerate(reactions): # loop over reactions
        # Find product of input species densities and multiply by reaction coefficient
        rate = drc[rx] * np.array([n[:,js] for js in nin[rx]]).prod(axis=0)
        # Compute change in densities for each reaction
        for js in nin[rx]:
            dndt[:,js] -= rate
        for js in nout[rx]:
            dndt[:,js] += rate
        for je in nemis[rx]:
            ver[:,je] += rate
        # Save stuff for diagnostics, as needed
        r[:,jr] = rate
        for js in nin[rx]:
            dndtfull[:,jr,js] -= rate
        for js in nout[rx]:
            dndtfull[:,jr,js] += rate
        for je in nemis[rx]:
            verfull[:,jr,je] += rate
    
    return r, dndt, ver, dndtfull, verfull

