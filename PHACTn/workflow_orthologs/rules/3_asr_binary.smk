rule binary_asr:
    input:
        msa_binary = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta_Binary",
        ancestralTree = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}.raxml.ancestralTree",
    params:
        raxml_ancestral_out_name = "{workdir}/{score_result_dirname}/{query_id}/3_binary_raxmlng_ancestral/{query_id}",
    output:
        ancestral_probabilities_binary = "{workdir}/{score_result_dirname}/{query_id}/3_binary_raxmlng_ancestral/{query_id}.raxml.ancestralProbs",
        ancestralTree_binary = "{workdir}/{score_result_dirname}/{query_id}/3_binary_raxmlng_ancestral/{query_id}.raxml.ancestralTree",
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/binary_asr.err" 
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/binary_asr.err" 
    resources:
        cpus=4
    conda:
        "../conda_env.yml"
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        raxml-ng --ancestral \
            --msa {input.msa_binary} \
            --tree {input.ancestralTree} \
            --model BIN \
            --opt-branches off \
            --prefix {params.raxml_ancestral_out_name} \
            --force perf_threads --seed {config[raxml_seed]} &&
        echo "`date -R`: {rule} ended successfully!" ||
        {{ echo "`date -R`: {rule} failed..."; exit 1; }} ) >> {log} 2>&1

        if [ ! -s {output.ancestral_probabilities_binary} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        if [ ! -s {output.ancestralTree_binary} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """
