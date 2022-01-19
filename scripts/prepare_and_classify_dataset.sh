#!/bin/bash

# Pipeline:
# - split reads, merge them if they can be merged. into a fasta file
# - see if they have other letters of the alphabet if N, replace by other letter
# - do same to protein files.
# Convention: processed_genome.seq; processed_protein.seq 

PROTEIN_OUT(){
    proteinfile="${1}""/protein.faa"
    if [[ -f ${proteinfile} ]]; then
        processed_files="${1}""/output_protein";
        # rm ${1}"/processed_protein.seq";
        if [[ ! -f ${1}"/processed_protein.seq" ]]; then
            rm -f "${processed_files}"/HEADER "${processed_files}"/PROTEIN;
            mkdir -p "${processed_files}";
            gto_fasta_split_reads -l "${processed_files}" < "${proteinfile}";
            head -n 1 "${proteinfile}" > "${processed_files}/"HEADER;
            cd ${processed_files};
            for file in ./*".fasta";  do
                # echo $file
                tail -n +2 "$file" >> PROTEIN;
            done
            cat HEADER PROTEIN > FILE;
            gto_fasta_to_seq < FILE > "../processed_protein.seq"
            rm *
            cd ..
            rmdir "output_protein"
            cd "../../../scripts";
        fi
    fi
}
GEN_OUT(){
    # rm ${2}"/processed_genome.seq";
    if [[ ! -f ${1}"/processed_genome.seq" ]]; then
        processed_files="${2}""/output_genome"
        rm -f "${processed_files}"/HEADER "${processed_files}"/FASTA
        mkdir -p ${processed_files};
        gto_fasta_split_reads -l "${processed_files}" < $1;
        head -n 1 "$1" > "${processed_files}/"HEADER;
        cd ${processed_files};
        for file in ./*".fasta";  do
            echo $file
            tail -n +2 "$file" > FST;
            n_symb=$(gto_info < FST |grep "Alphabet size" |awk -F ":" '{print $2}');
            if (( n_symb > 5 )); then
                gto_fasta_rand_extra_chars < "$file" > new_file;
                tail -n +2 new_file > FST
            fi
            cat FST >> FASTA
        done
        cat HEADER FASTA > FILE;
        gto_fasta_to_seq < FILE > "../processed_genome.seq"
        rm *
        cd ..
        rmdir "output_genome"
        cd "../../../scripts";
    fi
}


PREPROCESS_FILES(){
    for directory in  ../ncbi_dataset/data/*; do
        if [ -d "${directory}" ]; then
            dir=$(basename -- "${directory}")
            genome_file=$(find "${directory}" -name "${dir}"*".fna");
            if [[ ! -f ${genome_file} ]]; then
                genome_file=$(find "${directory}" -name "cds_from_genomic.fna");
            fi
            PROTEIN_OUT "${directory}"
            GEN_OUT "${genome_file}" "${directory}"
        fi
    done
}

GenomeInfo () {
    touch ../taxonomic_info/ArcheaSeq_Org.info ;
    for directory in  ../ncbi_dataset/data/*; do
        if [ -d "${directory}" ]; then
            dir=$(basename -- "${directory}")
            GenBank_Accn=$(cat "${directory}""/sequence_report.jsonl" | jq -r '.genbankAccession' )
            if ! ( < ../taxonomic_info/ArcheaSeq_Org.info grep -q "$dir".fasta);then
                esearch -db nuccore -query "$GenBank_Accn [ACCN]" | efetch -format gpc > "$1"
                taxonomy_name=$(cat "$1" | xtract -insd INSDSeq_taxonomy| awk -F '\t' '{print $2}')
                organism=$(cat "$1" | xtract -insd INSDSeq_organism| awk -F '\t' '{print $2}')
                echo -e "$dir.fasta\t${organism}\t${taxonomy_name}" >> ../taxonomic_info/ArcheaSeq_Org.info; 
            fi
        fi
    done
    rm "$1"
}

function SEQ_FEATURES(){
    mkdir -p "../reports";
    rm -f ../reports/"$1";
    for directory in  ../ncbi_dataset/data/*; do
        if [ -d "${directory}" ]; then

        genome_file="${directory}/processed_genome.seq";
        protein_file="${directory}/processed_protein.seq";
        if [[ -f "${protein_file}" ]]  && [[ -f "$genome_file" ]] ; then
            f="$(basename -- "$directory")";
            name="$f";
            len_x=$(wc -m <"$genome_file");
            gto_genomic_count_bases < "$genome_file" > GCTA; 
            nbases=$(sed "2q;d" GCTA | awk -F ":" '{print $2}');
            nA=$(sed "3q;d" GCTA | awk -F ":" '{print $2}');
            nC=$(sed "4q;d" GCTA | awk -F ":" '{print $2}');
            nG=$(sed "5q;d" GCTA | awk -F ":" '{print $2}');
            nT=$(sed "6q;d" GCTA | awk -F ":" '{print $2}');
            
            A_p=$(echo "scale=10; (${nA}) / ${nbases}" | bc -l | awk '{printf "%f", $0}');
            C_p=$(echo "scale=10; (${nC}) / ${nbases}" | bc -l | awk '{printf "%f", $0}');
            G_p=$(echo "scale=10; (${nC}) / ${nbases}" | bc -l | awk '{printf "%f", $0}');
            T_p=$(echo "scale=10; (${nT}) / ${nbases}" | bc -l | awk '{printf "%f", $0}');

            GC_p=$(echo "scale=10; (${nC}+${nG}) / ${nbases}" | bc -l | awk '{printf "%f", $0}');
            GeCo3 -v -l 3 "$genome_file" 1> report_stdout_nc 2> report_stderr_nc
            BPS1=$(grep "Total bytes" report_stdout_nc | awk '{ print $6; }');
            entropy=$(echo "scale=10; ($BPS1) / 2" | bc -l | awk '{printf "%f", $0}');
            len_p=$(wc -m <"$protein_file")
            a=$(AC -v -l 3 "$protein_file" | sed '3,4d' | sed 'N;s/\n/ /')
            ndc=$(echo $a | cut -d ' ' -f16);
            echo -e "${name}\t${len_x}\t${A_p}\t${C_p}\t${G_p}\t${T_p}\t${GC_p}\t${entropy}\t${len_p}\t${ndc}" >> ../reports/"$1"
            fi
        fi
    done
    rm GCTA report_stdout_nc report_stderr_nc
}

# Get NC, Sequence Length and GC-Content.
# Classification

##### Code
DATASET_DIR="../ncbi_dataset"
if [ ! -d "$DATASET_DIR" ]; then echo "Please run download_dataset.sh"; exit; fi
PREPROCESS_FILES
SEQ_FEATURES REPORTS_SEQ_FEATURES
GenomeInfo "A"
cd "../python_src/" || exit
python3 create_classification_dataset.py
python3 classification.py