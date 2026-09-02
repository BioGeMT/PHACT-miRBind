rule binary_asr:
    input:
        msa_binary = "{workdir}/{result_dirname}/{query_id}/1_preprocessed/block-{query_id}.fasta_Binary",
        ancestralTree = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.treefile",
    output:
        ancestral_probabilities_binary = "{workdir}/{result_dirname}/{query_id}/3_binary_iqtree_ancestral/{query_id}.state.gz",
        ancestralTree_binary = "{workdir}/{result_dirname}/{query_id}/3_binary_iqtree_ancestral/{query_id}.treefile",
    params:
        iqtree_ancestral_out_name = "{workdir}/{result_dirname}/{query_id}/3_binary_iqtree_ancestral/{query_id}",
    log:
        "{workdir}/logs/{result_dirname}/rules/{query_id}_binary_asr.err"
    benchmark:
        "{workdir}/logs/{result_dirname}/benchmarks/{query_id}_binary_asr.out"
    resources:
        cpus=8
    conda:
        "../envs/asr.yml"
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        iqtree2 -asr \
        -s {input.msa_binary} -te {input.ancestralTree} \
        -blfix -m JC2 --safe \
        --prefix {params.iqtree_ancestral_out_name} --redo &&

        gzip -c {params.iqtree_ancestral_out_name}.state > {output.ancestral_probabilities_binary} &&
        rm {params.iqtree_ancestral_out_name}.state &&

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