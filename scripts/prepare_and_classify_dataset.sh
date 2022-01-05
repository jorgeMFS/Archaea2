#!/bin/bash

REMOVE_HEADERS(){
    mkdir -p ../NCBI-Archaea/noheader/
    for directory in  ../NCBI-Archaea/ncbi_dataset/data/*; do
        if [ -d "${directory}" ]; then
            dir=$(basename -- "${directory}")
            for file in "${directory}""/"*".fna"; do
                head -n 1 "$file" > HEADER;
                tail -n +2 "$file" > FASTA
                if grep -q "N" FASTA; then
                    gto_fasta_rand_extra_chars < "$file" > A;
                    tail -n +2 A > FASTA
                fi
            
                header_l=$(wc -l HEADER | awk '{print $1}');
                fasta_l=$(wc -l FASTA |  awk '{print $1}');
                file_l=$(wc -l "$file"  |  awk '{print $1}');
                cat HEADER FASTA > FILE;
                sum=$(( $header_l + $fasta_l ))
                if [ "$sum" -eq "$file_l" ]; then
                    gto_fasta_to_seq < FILE > "../NCBI-Archaea/noheader/""${dir}"".fasta"
                else
                    cat HEADER FASTA
                    echo "-------------"
                    cat "$file"
                    echo "-------------"
                    cat A;
                    echo "-------------"
                    echo "wc sum : $sum, wc file: $ $file_l"            
                    echo "ERROR";
                    exit;
                fi 
            done
        fi
    done
    rm FASTA HEADER FILE
}

GenomeInfo () {
    touch ../NCBI-Archaea/ArcheaSeq_Org.info ;
    for directory in  ../NCBI-Archaea/ncbi_dataset/data/*; do
        if [ -d "${directory}" ]; then
            dir=$(basename -- "${directory}")
            GenBank_Accn=$(cat "${directory}""/sequence_report.jsonl" | jq -r '.genbankAccession' )
            if ! ( < ../NCBI-Archaea/ArcheaSeq_Org.info grep -q "$dir".fasta);then
                esearch -db nuccore -query "$GenBank_Accn [ACCN]" | efetch -format gpc > "$1"
                taxonomy_name=$(cat "$1" | xtract -insd INSDSeq_taxonomy| awk -F '\t' '{print $2}')
                organism=$(cat "$1" | xtract -insd INSDSeq_organism| awk -F '\t' '{print $2}')
                echo -e "$dir.fasta\t${organism}\t${taxonomy_name}" >> ../NCBI-Archaea/ArcheaSeq_Org.info; 
            fi
        fi
    done
    rm "$1"
}

function SEQ_FEATURES(){
    mkdir -p ../reports

    rm -f ../reports/"$1";
    for file in "../NCBI-Archaea/noheader/"*".fasta"; do
        echo "Running $file...";
        f="$(basename -- "$file")"
        name=$f
        len_x=$(wc -m <"$file")
        gto_genomic_count_bases < "$file" > GCTA; 
        
        nbases=$(sed "2q;d" GCTA | awk -F ":" '{print $2}')
        nC=$(sed "4q;d" GCTA | awk -F ":" '{print $2}')
        nG=$(sed "5q;d" GCTA | awk -F ":" '{print $2}')
        GC_p=$(echo "scale=10; (${nC}+${nG}) / ${nbases}" | bc -l | awk '{printf "%f", $0}');
        GeCo3 -v -l 3 "$file" 1> report_stdout_nc 2> report_stderr_nc
        BPS1=$(grep "Total bytes" report_stdout_nc | awk '{ print $6; }');
        entropy=$(echo "scale=10; ($BPS1) / 2" | bc -l | awk '{printf "%f", $0}')
        echo -e "$name\t$len_x\t$GC_p\t$entropy" >> ../reports/"$1"
    done
    rm GCTA report_stdout_nc report_stderr_nc
}


# Get NC, Sequence Length and GC-Content.
# Classification

##### Code
DATASET_DIR="../NCBI-Archaea/ncbi_dataset"
rm ../NCBI-Archaea/README.md > /dev/null
if [ -d "$DATASET_DIR" ]; then rm -Rf $DATASET_DIR; fi
cd "../NCBI-Archaea/" || exit
unzip "archaea.zip"
REMOVE_HEADERS
GenomeInfo "A"
SEQ_FEATURES REPORTS_SEQ_FEATURES
cd "../python_src/" || exit
python3 create_classification_dataset.py
python3 classification.py