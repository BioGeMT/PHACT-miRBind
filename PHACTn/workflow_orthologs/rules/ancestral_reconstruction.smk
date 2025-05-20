rule iqtree_ancestral:
    input:
        msa_file_nogap = "{workdir}/{score_result_dirname}/{query_id}/1_processed_msa/{query_id}_converted_nogap.fasta",
        bestTree_unrooted = "{workdir}/{score_result_dirname}/{query_id}/2_unrooted_tree/{query_id}_precursor_annotated.treefile_unrooted",
    output:
        ancestral_probabilities = "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral/{query_id}.state",
        ancestralTree = "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral/{query_id}.treefile",
    params:
        iqtree_ancestral_out_name = "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral/{query_id}",
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/iqtree_ancestral.err",
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/iqtree_ancestral.out"
    conda:
        "../../conda_env.yml"
    resources:
        cpus=4
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        iqtree2 -redo -s {input.msa_file_nogap}\
         -te {input.bestTree_unrooted}\
         -m {config[iqtree_ancestral_model]}\
         -asr --seqtype DNA --seed {config[iqtree_seed]}\
         --safe -nt {resources.cpus} --prefix {params.iqtree_ancestral_out_name} &&
        

        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output.ancestral_probabilities} ]; then
            echo "`date -R`: ancestral_probabilities is empty. Exiting..." >> {log}
            exit 1
        fi

        if [ ! -s {output.ancestralTree} ]; then
            echo "`date -R`: ancestralTree is empty. Exiting..." >> {log}
            exit 1
        fi

        """
    
