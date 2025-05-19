
rule iqtree_score:
    input:
        ancestral_probabilities = "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral/{query_id}.state",
        ancestralTree = "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral/{query_id}.treefile",
    output:
        "{workdir}/{score_result_dirname}/{query_id}/4_iqtree_ancestral_scores/{query_id}_wol_param_{pattern}.csv",
        "{workdir}/{score_result_dirname}/{query_id}/4_iqtree_ancestral_scores/{query_id}_wl_param_{pattern}.csv",
    params:
        out = "{workdir}/{score_result_dirname}/{query_id}/4_iqtree_ancestral_scores/{query_id}",
        msa_fasta = "{workdir}/{score_result_dirname}/{query_id}/1_processed_msa/{query_id}_converted_nogap.fasta",
        query = "{query_id}"
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/iqtreeanc_{pattern}_compute_score.err"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/iqtreeanc_{pattern}_compute_score.out"
    conda:
        "../../conda_env.yml"
    resources:
        cpus=4
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        Rscript scripts/computescores.R\
         {input.ancestralTree}\
         {input.ancestral_probabilities}\
         {params.msa_fasta} {params.out}\
         {params.query} {config[weights]} &&

        
        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """
