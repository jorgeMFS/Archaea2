#!/bin/bash

Check_Installation() {
    if ! [ -x "$(command -v "$1")" ]; then
        echo -e "\033[6;31mERROR\033[0;31m: $1 is not installed!" >&2;
        exit 1;
    else
        echo -e "\033[1;32mSUCCESS!\033[0m";
    fi
}

echo "Start Installation..."

conda install -c cobilab gto --yes
conda install -y -c bioconda geco3 --yes
conda install -c bioconda ac --yes
conda install -c bioconda entrez-direct --yes
conda install -c https://conda.anaconda.org/biocore scikit-bio  --yes

#
pip install openpyxl
pip install pandas
pip install numpy
pip install sklearn
pip install xgboost


Check_Installation "gto";
Check_Installation "GeCo3";
Check_Installation "AC";
Check_Installation "efetch";