
rule compute_scores_gapAware:
    input:
        ancestral_probabilities_posterior = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}_gapAware_probabilities.tsv",
        ancestralTree = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}.raxml.ancestralTree",
        msa_fasta = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta",
    output:
        wol = "{workdir}/{score_result_dirname}/{query_id}/PHACTn_gapAware_scores_{nt_norm}/{query_id}_wol_param_{pattern}.csv",
        wl = "{workdir}/{score_result_dirname}/{query_id}/PHACTn_gapAware_scores_{nt_norm}/{query_id}_wl_param_{pattern}.csv",
    params:
        out_prefix = "{workdir}/{score_result_dirname}/{query_id}/PHACTn_gapAware_scores_{nt_norm}/{query_id}",
        query = "Hsa"
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}_gapAware_{nt_norm}_scores_{pattern}.log"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}_gapAware_{nt_norm}_scores_{pattern}.out"
    conda:
        "../conda_env.yml"
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
