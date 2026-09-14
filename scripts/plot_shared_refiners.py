#!/usr/bin/env python3
"""Measured shared-refiner cost and risk limits; reads audited analysis only."""
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/agent_shared_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/shared_refiner_20260909'


def main():
    report=json.loads((OUT/'analysis.json').read_text())
    audit=json.loads((OUT/'independent_audit.json').read_text())
    conditional=json.loads((OUT/'conditional_prefix_analysis.json').read_text())
    assert audit['status']=='PASS'
    names=['standard','targeted','two_stage']
    pretty=['Standard','Targeted','Two-stage']
    blue='#245D96';orange='#C6672D';ink='#222222'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'text.color':ink,'axes.labelcolor':ink})
    fig,ax=plt.subplots(1,2,figsize=(6.5,3.1),gridspec_kw={'width_ratios':[1.2,1]})
    upstream=report['reference_plus_gate']['tokens']/1e6
    extras=np.array([0]+[report['refiners'][a]['cost']['tokens']/1e6 for a in names]+[report['total_measured_refinement']['tokens']/1e6])
    x=np.arange(5)
    ax[0].bar(x,[upstream]*5,color=blue,width=.65,label='Reference + initial verifier')
    ax[0].bar(x,extras,bottom=upstream,color=orange,width=.65,label='Measured refinement')
    for i,v in enumerate(extras):ax[0].text(i,upstream+v+.035,f'{upstream+v:.3f}',ha='center',va='bottom',fontsize=7)
    ax[0].set_xticks(x,['FRR\nonly','Direct:\nstandard','Direct:\ntargeted','Direct:\ntwo-stage','Direct:\nall three'],fontsize=7)
    ax[0].set_ylim(0,(upstream+max(extras))*1.25)
    ax[0].set_ylabel('Recorded tokens (millions)')
    ax[0].set_title('(a) Recorded model-token cost',loc='left')
    ax[0].legend(loc='upper left',fontsize=7.5,frameon=False)
    ax[0].grid(axis='y',alpha=.15);ax[0].set_axisbelow(True)
    ceiling=conditional['deterministic_shared_ceiling']
    upper=[conditional['refiners'][a]['direct_simultaneous_upper'] for a in names]
    ax[1].scatter([0],[ceiling],c=blue,marker='D',s=32,label='Exact shared ceiling on stored prefix',zorder=3)
    ax[1].scatter(np.arange(1,4),upper,c=orange,marker='s',s=28,label='Direct, simultaneous Hoeffding UCB',zorder=3)
    ax[1].axhline(.05,color='#555555',linestyle='--',linewidth=.9)
    ax[1].text(3.48,.051,'5% primary',ha='right',fontsize=7,color='#555555')
    ax[1].axhline(.02,color='#888888',linestyle=':',linewidth=.8)
    ax[1].text(3.48,.021,'2% secondary',ha='right',fontsize=6.5,color='#777777')
    ax[1].set_xticks(np.arange(4),['Shared\nceiling']+[f'{a}\ndirect' for a in pretty],fontsize=7)
    ax[1].set_ylim(0,.072);ax[1].set_xlim(-.5,3.5)
    ax[1].set_ylabel('Upper bound on breakage')
    ax[1].set_title('(b) Risk on the stored prefix',loc='left')
    ax[1].legend(loc='upper left',fontsize=7.5,frameon=False)
    ax[1].grid(axis='y',alpha=.15);ax[1].set_axisbelow(True)
    fig.subplots_adjust(left=.07,right=.99,bottom=.2,top=.88,wspace=.32)
    fig.savefig(ROOT/'paper/fig_shared_refiners.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/'fig_shared_refiners.png',dpi=200,bbox_inches='tight',pad_inches=.03)
    print('Created measured shared-refiner figure.')


if __name__=='__main__':main()
