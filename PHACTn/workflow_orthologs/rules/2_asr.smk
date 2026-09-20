
rule ancestral_reconst:
    input:
        msa_file_nogap = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta",
        besttree_unrooted = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}.bestTree_unrooted",
    params:
        raxmlng_ancestral_out_name = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}",
    output:
        ancestral_probabilities = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}.raxml.ancestralProbs",
        ancestralTree = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}.raxml.ancestralTree",
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/raxmlng_ancestral.err",
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/raxmlng_ancestral.out"
    conda:
        "../conda_env.yml"
    resources:
        cpus=4
    shell:
        """ 
        (
         echo "`date -R`: {rule} started..." &&


        raxml-ng --ancestral \
         --msa {input.msa_file_nogap} --tree {input.besttree_unrooted} \
         --model {config[raxml_model]} --prefix {params.raxmlng_ancestral_out_name} \
         --threads {resources.cpus} --seed {config[raxml_seed]} --extra seq-allgap-keep --extra seq-dup-keep &&


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
    
