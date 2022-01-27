#!/bin/bash
#

Check_Installation() {
    if ! [ -x "$(command -v "$1")" ]; then
        echo -e "\033[6;31mERROR\033[0;31m: $1 is not installed!" >&2;
        exit 1;
    else
        echo -e "\033[1;32mSUCCESS!\033[0m";
    fi
}


echo -e "\033[1mStart Tool Installation...\033[0m"
#TODO

conda install -y -c bioconda geco3
Check_Installation "GeCo3";
conda install -c bioconda entrez-direct --yes
Check_Installation "efetch";
conda install -c cobilab gto --yes 
Check_Installation "gto";
conda install -c bioconda ac --yes
Check_Installation "AC";
conda install -c conda-forge ncbi-datasets-cli


echo -e "\033[1;32mSuccessfully installed tools!\033[0m";