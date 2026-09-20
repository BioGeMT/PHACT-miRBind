rule processMSA:
    input:
        msa_file = lambda wildcards: f"{config['msa_file_pattern']}".format(query_id=wildcards.query_id),
        tree = lambda wildcards: f"{config['tree_file_path']}",
    output:
        msa_filtered = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/block-{query_id}_noGapped.fasta",
        pruned_tree = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/prunedtree_{query_id}.nwk",
    log:
        "{workdir}/logs/{result_dirname}/rules/{query_id}_gapped_filtered.err"
    benchmark:
        "{workdir}/logs/{result_dirname}/benchmarks/{query_id}_gapped_filtered.out"
    conda:
        "../envs/core.yml"
    resources:
        cpus=4
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        python3 process_MSA.py \
         --input_msa {input.msa_file} \
         --input_tree {input.tree} \
         --output_msa {output.msa_filtered} \
         --output_tree {output.pruned_tree} &&

        echo "`date -R`: {rule} ended successfully!" ||
        {{ echo "`date -R`: {rule} failed..."; exit 1; }} ) >> {log} 2>&1

        if [ ! -s {output.msa_filtered} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        
        if [ ! -s {output.pruned_tree} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi

        """

rule unroot_pruned:
    input:
        pruned_tree = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/prunedtree_{query_id}.nwk"
    output:
        "{workdir}/{result_dirname}/{query_id}/1_preprocessed/prunedtree_{query_id}.nwk_unrooted"
    log:
        "{workdir}/logs/{result_dirname}/rules/{query_id}_unroot.err"
    benchmark:
        "{workdir}/logs/{result_dirname}/benchmarks/{query_id}_unroot.out"
    conda:
        "../envs/core.yml"
    resources:
        cpus=4
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        Rscript ../scripts/unroot_tree.R {input.pruned_tree} {output} &&

        echo "`date -R`: {rule} ended successfully!" ||
        {{ echo "`date -R`: {rule} failed..."; exit 1; }} ) >> {log} 2>&1

        if [ ! -s {output} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """

