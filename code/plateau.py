import pandas as pd
#import matplotlib.pyplot as plt

def get_slope(my_list):
    x = range(len(my_list))
    y = my_list
    slope = (y[-1]-y[0])/(x[-1]-x[0])
    return slope

def get_plateau(my_list, window=10, criterion=20):
    '''return index when no improvement for n_steps steps'''
    my_list = list(pd.Series(my_list).rolling(window=window,min_periods=1).apply(lambda x: sum(x)/window))
    my_slopes = list(pd.Series(my_list).rolling(window=2,min_periods=1).apply(get_slope))
    max_slope = max(my_slopes[1:])
    for idx,slope in enumerate(my_slopes):
        if slope < max_slope / criterion:
            best_idxx = idx
            break
    if 'best_idxx' not in locals():
        best_idxx = idx
    return best_idxx, my_list


#metric = [0.1,0.2,0.25,0.4,0.65,0.8,0.85,0.86,0.87]
#plt.plot(range(len(metric)),metric)
#plt.axvline(x=get_plateau(metric,2),label='plateau')
#plt.show()





