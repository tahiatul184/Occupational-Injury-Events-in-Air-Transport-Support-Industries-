"""Plot the published aggregate confusion counts, without report-level inputs."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'local_outputs/figures')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for split,title in [('SIR_2022_temporal_test','SIR 2022 published codes'),
                        ('ITA_2023_autocoder_comparison','ITA 2023 autocoder references'),
                        ('ITA_2024_autocoder_comparison','ITA 2024 autocoder references')]:
        matrix=pd.read_csv(ROOT/'results'/(split+'_tfidf_logistic_confusion.csv'),index_col=0).to_numpy()
        fig,ax=plt.subplots(figsize=(6.4,5.5))
        ax.imshow(matrix,cmap='Blues',vmin=0)
        ax.set(xticks=range(7),yticks=range(7),xticklabels=list('1234567'),yticklabels=list('1234567'),
               xlabel='Model predicted division',ylabel='Source reference division',title=title+'   n = '+str(matrix.sum()))
        for y in range(7):
            for x in range(7):ax.text(x,y,str(matrix[y,x]),ha='center',va='center',color='white' if matrix[y,x]>matrix.max()*.53 else '#333333')
        ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,7,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=.6);ax.tick_params(which='minor',bottom=False,left=False)
        fig.tight_layout();fig.savefig(args.output_dir/(split+'.png'),dpi=300,facecolor='white');plt.close(fig)
    print('Three aggregate confusion plots written to '+str(args.output_dir))
if __name__=='__main__':main()
