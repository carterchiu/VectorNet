''' Produces a configuration diagram from a configuration string (e.g., "| E(5,NID) -- | -- A(MED) | E(2,ID) -- | -- A(MED) | -- -- |" '''

import matplotlib.pyplot as plt
import matplotlib.patches as pat
import sys

init, label = sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None
    
layers = sum([i=='|' for i in list(init)]) - 1

comps = [i for i in init.split(' ') if i != '|']

cen = 1.625
fig, ax = plt.subplots(1,1, figsize=(layers * 2 + 1.5, cen * 2 if not label else cen * 2.25))
ax.set_xlim([-.5, layers * 2 + 2 - .5])
ax.set_ylim([0, cen * 2 if not label else cen * 2.25])
ax.axis('off')
height = 1

if label:
    ax.text(0.25, cen * 2.125, label, c='black', fontsize=18, fontname='Nexa Light')

for i in range(layers):
    
    expstr, aggstr = comps[2 * i], comps[2 * i + 1]
    
    ax.plot([2 * i + .25] * 2, [0, cen * 2], ls='--', c='#7A7A7A', lw=1, alpha=.5)
    if height == 1:
        ax.add_patch(pat.Circle((2 * i + .25, cen), .25, color='#D4DCE2', ec='#AABAC6'))
    else:
        ax.add_patch(pat.FancyBboxPatch((2 * i + .25, cen - .25 * height + .25), 0, .5 * height - .5, boxstyle='round, pad=.25', color='#D4DCE2', ec='#AABAC6'))
    start, end = 2 * i + .5, 2 * i + 2
    
    if expstr != '--':
        e_fct = expstr[expstr.index(',')+1:expstr.index(')')]
        e_size = int(expstr[expstr.index('(')+1:expstr.index(',')])
        height = e_size
        start = 2 * i + 1.125
        for h in range(height):
            ax.plot([2 * i + .25, 2 * i + .875],[cen, cen + .25 - .25 * height + h / 2], color='#1F77B4', lw=2, ls='--', dashes=[2, 1], zorder=-1)
        ax.text(2 * i + .5625, cen - .625, e_fct, color='w', bbox=dict(facecolor='#1F77B4', lw=0), ha='center')
        ax.add_patch(pat.FancyBboxPatch((2 * i + .875, cen - .25 * height + .25), 0, .5 * height - .5, boxstyle='round, pad=.25', color='#D4DCE2', ec='#AABAC6'))
    
    if aggstr != '--':
        a_fct = aggstr[aggstr.index('(')+1:aggstr.index(')')]
        for h in range(height):
            ax.plot([2 * i + 1.625, 2 * i + 2.25],[cen + .25 - .25 * height + h / 2, cen], color='#FF7F0E', lw=2, ls='--', dashes=[2, 1], zorder=-1)
        ax.add_patch(pat.FancyBboxPatch((2 * i + 1.625, cen - .25 * height + .25), 0, .5 * height - .5, boxstyle='round, pad=.25', color='#D4DCE2', ec='#AABAC6'))
        height = 1
        end = 2 * i + 1.375
        ax.text(2 * i + 1.9375, cen - .625, a_fct, color='w', bbox=dict(facecolor='#FF7F0E', lw=0), ha='center')

    ax.plot([start, end], [cen, cen], color='black', lw=1.25, zorder=-1)
    ax.text(2 * i + 1.25, .5, i + 1, c='#7A7A7A', ha='center')
ax.plot([2 * layers + .25] * 2, [0, cen * 2], ls='--', c='#7A7A7A', lw=1, alpha=.5)
ax.add_patch(pat.Circle((2 * layers + .25, cen), .25, color='#D4DCE2', ec='#AABAC6'))