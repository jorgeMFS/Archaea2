#!python
#!/usr/bin/env python

import sys
import numpy as np

def read_to_list(path):
    organism_list=[]
    with open(path, 'r') as file:
        data = file.read()
    data = data.splitlines()
    list_of_lists=[]
    for line in data:
        line = line.split('\t')
        line = [s.strip() for s in line]
        list_of_lists.append(line)
    return list_of_lists

def process_feature_lst(lst):
    return [[x[0],float(x[1]),float(x[2]),float(x[3])]for x in lst] 

def restructure_organism_list(lst):
    return [[x[0]]+x[2].split("; ") for x in lst]

def merge_lists(lst1,lst2):
    return [ y[1:]+x[1:] for x in lst1 for y in lst2 if x[0]==y[0]]

def classification_structure(lst,map_lst):
    tax=["Phylum","Class","Order","Family","Genus"]
    new_lst=[]
    for tx in tax:
        clss_lst=[]
        labels=[]
        lst_tx=[x[0] for x in map_lst if x[1]==tx]
        for l_tx, cnt in zip(lst_tx, list(range(len(lst_tx)))):
            for x in lst:
                if l_tx in x:
                    clss_lst.append(x[:3])
                    labels.append(cnt)
        
        clss_lst=np.array(clss_lst)
        labels=np.array(labels).astype('int32')
        np.save("../data/"+tx+'_'+'y_data.npy', labels)
        np.save("../data/"+tx+'_'+'x_data.npy', clss_lst)


if __name__ == "__main__": 
    features_path="../reports/REPORTS_SEQ_FEATURES"
    organism_info_path="../NCBI-Archaea/ArcheaSeq_Org.info"
    taxa_map_path="../NCBI-Archaea/taxa_map.info"
    organism_info=restructure_organism_list(read_to_list(organism_info_path))
    features=process_feature_lst(read_to_list(features_path))
    archea_lst=merge_lists(organism_info,features)
    taxa_map=read_to_list(taxa_map_path)
    organism_info=classification_structure(archea_lst,taxa_map)
    

