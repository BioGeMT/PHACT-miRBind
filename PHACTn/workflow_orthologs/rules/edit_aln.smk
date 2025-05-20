import os
rule edit_alignment:
    input:
        msa_file = lambda wildcards: f"{config['alignment_pattern']}".format(query_id=wildcards.query_id),
    output:
        "{workdir}/{score_result_dirname}/{query_id}/1_processed_msa/{query_id}_converted_nogap.fasta"
    params:
        out_file = "{workdir}/{score_result_dirname}/{query_id}/1_processed_msa/{query_id}_converted.fasta"
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/msa_processing.err"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/msa_processing.out"
    conda:
        "../../conda_env.yml"
    resources:
        cpus=4
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        python3 scripts/edit_aln.py \
            -msa {input.msa_file} \
            -id {wildcards.query_id} \
            -out {params.out_file} &&

        
        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output} ]; then
            echo "`date -R`: output is empty. Exiting..." >> {log}
            exit 1
        fi
        """