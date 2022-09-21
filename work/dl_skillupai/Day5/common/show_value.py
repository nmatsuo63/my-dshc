import numpy as np
import matplotlib.pyplot as plt
from common.meiro import State


def show_q_value(Q, row, col):
    stride = 3 
    row_graph = row *stride
    col_graph = col *stride
    qmap = np.empty((row_graph, col_graph))
    qmap[:] = np.nan

    for r in range(row):
        for c in range(col):
            q = Q[State(r,c)]

            if sum(q)==0:
                """
                通常セル以外の場合
                """
                continue

            qmap[r*stride+1-1, c*stride+1] = q[0] # UP
            qmap[r*stride+1+1, c*stride+1] = q[1]  # DOWN 
            qmap[r*stride+1, c*stride+1-1] = q[2]  # LEFT
            qmap[r*stride+1, c*stride+1+1] = q[3] # RIGHT

    import seaborn as sns
    plt.figure(figsize=(12,10))
    sns.heatmap(qmap, annot=True, linewidths=5, vmin=-1, vmax=1, cmap=sns.color_palette("Reds", 24))
    plt.title("Calculated Value")
    x = np.arange(col_graph)
    y = np.arange(row_graph)    
    plt.xticks([i+1.5 for i in x[::3]])
    plt.yticks([i+1.5 for i in y[::3]])    
    plt.show()        
    

def show_v_value(V, row, col):
    row_graph = row
    col_graph = col
    vmap = np.empty((row_graph, col_graph))
    vmap[:] = np.nan
   
    for r in range(row):
        for c in range(col):
            v = V[State(r,c)]
            
            if v==0:
                """
                通常セル以外の場合
                """
                continue

            vmap[r, c] = v

    import seaborn as sns
    plt.figure(figsize=(12,10))
    sns.heatmap(vmap, annot=True, linewidths=5, vmin=-1, vmax=1, cmap=sns.color_palette("Reds", 24))
    plt.title("Calculated Value") 
    plt.show()        