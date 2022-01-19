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
    return [[x[0],int(x[1]),float(x[2]),float(x[3]),float(x[4]),float(x[5]),float(x[6]),float(x[7]), int(x[8]), float(x[9])] for x in lst] 

def restructure_organism_list(lst):  
    return [[x[0]]+x[2].split("; ") for x in lst]

def merge_lists(lst1,lst2):
    return [ y[1:]+x[1:] for x in lst1 for y in lst2 if x[0].replace(".fasta", "")==y[0]]

def filter_insufficient_samples(lst, lst_tx, min_number_samples):
    cnt_lst=[]
    for tx in lst_tx:
        a=[tx,0]
        for x in lst:
            if tx in x:
                a[1]+=1
        cnt_lst.append(a)
    
    lst_tx = [x[0] for x in cnt_lst if x[1]>=min_number_samples]
    return lst_tx

def classification_structure(lst,map_lst):
    tax=["Phylum","Class","Order","Family","Genus"]
    #Species,  Unknown not considerered
    
    new_lst=[]
    for tx in tax:
        clss_lst=[]
        labels=[]
        lst_tx=[x[0] for x in map_lst if x[1]==tx]
        lst_tx=filter_insufficient_samples(lst,lst_tx,4)
        for l_tx, cnt in zip(lst_tx, list(range(len(lst_tx)))):
            for x in lst:
                if l_tx in x:
                    clss_lst.append(x[:9])
                    labels.append(cnt)
        clss_lst=np.array(clss_lst)
        labels=np.array(labels).astype('int32')
        np.save("../data/"+tx+'_'+'y_data.npy', labels)
        np.save("../data/"+tx+'_'+'x_data.npy', clss_lst)

def filter_non_read(organism_info, taxa_map):
    organism_info=[org[1:] for org in organism_info]
    organism_info = list(set([item for sublist in organism_info for item in sublist]))
    tax=[tx[0] for tx in taxa_map]
    remaining=[]
    for org in organism_info:
        if org not in tax:
            remaining.append(org)
    remaining.sort()
    remaining = [el for el in remaining if "unclassified" not in el]
    remaining = [el for el in remaining if "Archaea" not in el]


if __name__ == "__main__": 
    features_path="../reports/REPORTS_SEQ_FEATURES"
    organism_info_path="../taxonomic_info/ArcheaSeq_Org.info"
    taxa_map_path="../taxonomic_info/taxa_map.info"
    organism_info=restructure_organism_list(read_to_list(organism_info_path))
    taxa_map=read_to_list(taxa_map_path)    
    features=process_feature_lst(read_to_list(features_path))
    # unique_groups=list(set([x[1] for x in taxa_map]))

    archea_lst=merge_lists(organism_info,features)
    organism_info=classification_structure(archea_lst,taxa_map)
    
    

