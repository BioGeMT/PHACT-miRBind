rule preprocess:
    input:
        msa_file = lambda wildcards: f"{config['alignment_pattern']}".format(query_id=wildcards.query_id),
    params:
        query_id = lambda wildcards: f"{wildcards.query_id}{config['id_prefix']}",
        consensus_tree = config['consensus_tree'],
    output:
        nt_out = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta",
        binary_out = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta_Binary"
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/preprocess.err"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/preprocess.out"
    conda:
        "../conda_env.yml"
    resources:
        cpus=4
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        python3 preprocess.py \
            -msa {input.msa_file} \
            -id {params.query_id} \
            -tree {params.consensus_tree} \
            -out {output.nt_out} \
            -bin_out {output.binary_out} &&

        
        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output.nt_out} ] || [ ! -s {output.binary_out} ]; then
            echo "`date -R`: output is empty. Exiting..." >> {log}
            exit 1
        fi
        """


rule constrained_tree:
    input:
        complete_msa = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta",
    params:
        consensus_tree = config['consensus_tree'],
        out_name = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}",
    output:
        best_tree = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}.raxml.bestTree",
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/constrained_tree.err"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/constrained_tree.out"
    conda:
        "../conda_env.yml"
    resources:
        cpus=2
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&

        raxml-ng --search \
         --msa {input.complete_msa} \
         --tree-constraint {params.consensus_tree} --seed {config[raxml_seed]} \
         --prefix {params.out_name} --model {config[raxml_model]} --extra seq-allgap-keep --extra seq-dup-keep --redo &&


        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output} ]; then
            echo "`date -R`: output is empty. Exiting..." >> {log}
            exit 1
        fi
        """

rule unroot_tree:
    input:
        best_tree = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}.raxml.bestTree",
    output:
        best_tree_unrooted = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}.bestTree_unrooted",
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/unroot.err"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/unroot.out"
    conda:
        "../conda_env.yml"
    resources:
        cpus=2
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        Rscript ../scripts/unroot_tree.R\
         {input.best_tree} \
         {output.best_tree_unrooted} &&


        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output} ]; then
            echo "`date -R`: output is empty. Exiting..." >> {log}
            exit 1
        fi
        """
