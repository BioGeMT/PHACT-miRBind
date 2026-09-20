rule iqtree_ancestral:
    input:
        msa_filtered = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/block-{query_id}_noGapped.fasta",
        pruned_unrooted_tree = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/prunedtree_{query_id}.nwk_unrooted",
    output:
        ancestral_probabilities = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.state.gz",
        ancestralTree = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.treefile",
    params:
        iqtree_ancestral_out_name = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}",
    log:
        "{workdir}/logs/{result_dirname}/rules/{query_id}_iqtree_ancestral.err"
    benchmark:
        "{workdir}/logs/{result_dirname}/benchmarks/{query_id}_iqtree_ancestral.out"
    conda:
        "../envs/asr.yml"
    resources:
        cpus=8
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        iqtree2 -redo -s {input.msa_filtered}\
         -te {input.pruned_unrooted_tree} -m {config[iqtree_ancestral_model]}\
         -asr --seqtype DNA --seed {config[iqtree_seed]} --safe -nt {resources.cpus}\
         --prefix {params.iqtree_ancestral_out_name} &&

        gzip -c {params.iqtree_ancestral_out_name}.state > {output.ancestral_probabilities} &&
        rm {params.iqtree_ancestral_out_name}.state &&

        echo "`date -R`: {rule} ended successfully!" ||
        {{ echo "`date -R`: {rule} failed..."; exit 1; }} ) >> {log} 2>&1

        if [ ! -s {output.ancestral_probabilities} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi

        if [ ! -s {output.ancestralTree} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """