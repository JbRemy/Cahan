import os
import json
import numpy as np

from tabulate import tabulate

path_root = 'C:/Users/mvazirg/Dropbox/'
path_to_results = path_root + '/casa_2_results/new_jb/'
path_write = 'C:/Users/mvazirg/casa_2/best_configs/jb/'

write_best_configs_only = True

datasets = ['amazon_review_full_csv','yelp_review_full_csv','yahoo_answers_csv']
names = ['jb','antoine','jesse']
name_dataset = dict(zip(datasets,names))

best_configs = dict()
to_write = dict()

for dataset in datasets:
    print('\n = = = = =',dataset,'= = = = =')
    path_tmp = path_to_results + name_dataset[dataset]
    configs = [elt for elt in os.listdir(path_tmp) if '=' in elt]
    perfs = []
    sds = []
    for config in configs:
        print('* * *',config,'* * *')
        with open(path_tmp + '/' + config + '/' + dataset + '/res.json') as file:
            tmp = json.load(file)
        if 'final' in tmp:
            perfs.append(round(tmp['final'],3))
            sds.append(round(tmp['final_std'],3))
            print(perfs[-1])
        else: 
            print('experiment still running')
    best_configs[dataset] = configs[perfs.index(max(perfs))]
    
    to_write[dataset] = dict(zip(configs,zip(perfs,sds)))

with open(path_write + 'best_configs.json', 'w') as file:
    json.dump(best_configs, file, sort_keys=False, indent=4)

with open(path_write + 'grid_search_scores.json', 'w') as file:
    json.dump(to_write, file, sort_keys=False, indent=4)

for k,v in to_write.items():
    print(k)
    to_print = {' '.join(kk.split('_')[:2]): vv for kk,vv in v.items()}
    print(tabulate(to_print.items(),tablefmt='latex'))

if not write_best_configs_only:
    
    # baselines
    
    perf_sds = dict()
    
    for name in ['jb','jesse','antoine']:
        dataset_names = os.listdir(path_to_results + name + '/baseline/')
        for dn in dataset_names:
            if os.path.isfile(path_to_results + name + '/baseline/' + dn + '/res.json'):
                with open(path_to_results + name + '/baseline/' + dn + '/res.json', 'r', encoding='utf-8') as my_file:
                    res = json.load(my_file)
                
                if 'final' not in res:
                    final_accs = [[v for k,v in vv.items() if k=='val_acc'] for kk,vv in res.items() if kk != 'lr']
                    final_accs = [elt for sublist in final_accs for elt in sublist]
                    final = round(np.mean(final_accs),3)
                    final_std = round(np.std(final_accs),3)
                    print('\n = = = = =',dn,'= = = = =')
                    print(final,final_std)
                    perf_sds[dn] = (final,final_std)
    
    with open(path_write + 'baseline_scores.json', 'w') as file:
        json.dump(perf_sds, file, sort_keys=False, indent=4)
    
    # best models
    for name in ['jb','jesse']:
        dataset_names = os.listdir(path_to_results + name + '/best/')
        for dn in dataset_names:
            if os.path.isfile(path_to_results + name + '/baseline/' + dn + '/res.json'):
                with open(path_to_results + name + '/baseline/' + dn + '/res.json', 'r', encoding='utf-8') as my_file:
                    res = json.load(my_file)
                
                if 'final' not in res:
                    final_accs = [[v for k,v in vv.items() if k=='val_acc'] for kk,vv in res.items() if kk != 'lr']
                    final_accs = [elt for sublist in final_accs for elt in sublist]
                    final = round(np.mean(final_accs),3)
                    final_std = round(np.std(final_accs),3)
                    print('\n = = = = =',dn,'= = = = =')
                    print(final,final_std)

