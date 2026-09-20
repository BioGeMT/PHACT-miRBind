rule compute_scores_gapAware:
    input:
        ancestral_probabilities_posterior = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}_gapAware_probabilities.tsv",
        ancestralTree = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.treefile",
        msa_fasta = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/block-{query_id}_noGapped.fasta",
    output:
        wol = "{workdir}/{result_dirname}/{query_id}/4_gapAware_scores_{nt_norm}/{query_id}_wol_param_{pattern}.csv",
        wl = "{workdir}/{result_dirname}/{query_id}/4_gapAware_scores_{nt_norm}/{query_id}_wl_param_{pattern}.csv",
    params:
        out_prefix = "{workdir}/{result_dirname}/{query_id}/4_gapAware_scores_{nt_norm}/{query_id}",
        query = config["query_species"] 
    log:
        "{workdir}/logs/{result_dirname}/rules/{query_id}_gapAware_{nt_norm}_scores_{pattern}.log"
    benchmark:
        "{workdir}/logs/{result_dirname}/benchmarks/{query_id}_gapAware_{nt_norm}_scores_{pattern}.out"
    conda:
        "../envs/score_r.yml"
    resources:
        cpus=4
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        Rscript ../scripts/PHACTn_scripts/computescores_{wildcards.nt_norm}.R \
            {input.ancestralTree} \
            {input.ancestral_probabilities_posterior} \
            {input.msa_fasta} \
            {params.out_prefix} \
            {params.query} \
            {config[weights]} &&

        echo "`date -R`: {rule} ended successfully!" ||
        {{ echo "`date -R`: {rule} failed..."; exit 1; }} ) >> {log} 2>&1

        if [ ! -s {output.wol} ] || [ ! -s {output.wl} ]; then
            echo "`date -R`: One or more output files are empty. Exiting..." >> {log}
            exit 1
        fi
        """
