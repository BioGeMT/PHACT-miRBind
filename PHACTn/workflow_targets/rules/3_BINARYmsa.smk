rule binary_msa:
    input:
        msa_filtered = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/block-{query_id}_noGapped.fasta"
    output:
        msa_binary = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/block-{query_id}.fasta_Binary"
    log:
        "{workdir}/logs/{result_dirname}/rules/{query_id}_binary_msa.err"
    benchmark:
        "{workdir}/logs/{result_dirname}/benchmarks/{query_id}_binary_msa.out"
    resources:
        cpus=4
    conda:
        "../envs/core.yml"
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        python3 binary_msa.py --msa_file {input.msa_filtered} --binary_out {output.msa_binary} &&
        echo "`date -R`: {rule} ended successfully!" ||
        {{ echo "`date -R`: {rule} failed..."; exit 1; }} ) >> {log} 2>&1

        if [ ! -s {output.msa_binary} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """

