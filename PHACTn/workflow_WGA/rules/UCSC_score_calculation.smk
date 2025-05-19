rule edit_WGA:
    input:
        msa_file = lambda wildcards: f"{config['msa_file_pattern']}".format(query_id=wildcards.query_id),
        tree_file = lambda wildcards: f"{config['tree_file_path']}"
    output:
        msa_file_filtered = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered.fasta",
        tree_file_pruned = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_pruned.nwk",
    params:
        query_id = lambda wildcards: wildcards.query_id
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/remove_seqs_wGaps.err"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/remove_seqs_wGaps.out"
    conda:
        "../../conda_env.yml"
    resources:
        cpus=4
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        python3 scripts/edit_aln_UCSC.py \
            --input_msa {input.msa_file} \
            --input_tree {input.tree_file} \
            --output_msa {output.msa_file_filtered} \
            --output_tree {output.tree_file_pruned} \
            --query_id {params.query_id}\
            --positions_file scripts/hsa_pre_miRNA_positions.tsv &&


        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output.msa_file_filtered} ]; then
            echo "`date -R`: msa_file_filtered is empty. Exiting..." >> {log}
            exit 1
        fi

        if [ ! -s {output.tree_file_pruned} ]; then
            echo "`date -R`: tree_file_pruned is empty. Exiting..." >> {log}
            exit 1
        fi

        """

rule remove_gaps_UCSC:
    input:
        msa_file_filtered = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered.fasta",
    output:
        msa_file_nogap = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta",
    params:
        human_id = "hg38"
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/remove_gaps.err"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/remove_gaps.out"
    conda:
        "../../conda_env.yml"
    resources:
        cpus=4
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        python3 scripts/rm_gaps.py\
         -msa {input.msa_file_filtered} \
         -id {params.human_id} \
         -out {output.msa_file_nogap} &&


        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output.msa_file_nogap} ]; then
            echo "`date -R`: msa_file_nogap is empty. Exiting..." >> {log}
            exit 1
        fi
        """

rule iqtree_ancestral_UCSC:
    input:
        msa_file_nogap = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta",
        tree_file_pruned = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_pruned.nwk",
    output:
        ancestral_probabilities = "{workdir}/{score_result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.state",
        ancestralTree = "{workdir}/{score_result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.treefile",
    params:
        iqtree_ancestral_out_name = "{workdir}/{score_result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}",
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
         -te {input.tree_file_pruned}\
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
        

rule iqtree_score_UCSC:
    input:
        ancestral_probabilities = "{workdir}/{score_result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.state",
        ancestralTree = "{workdir}/{score_result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.treefile",
    output:
        "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral_scores/{query_id}_wol_param_{pattern}.csv",
        "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral_scores/{query_id}_wl_param_{pattern}.csv",
    params:
        out = "{workdir}/{score_result_dirname}/{query_id}/3_iqtree_ancestral_scores/{query_id}",
        msa_fasta = "{workdir}/{score_result_dirname}/{query_id}/1_preProcessing/{query_id}_filtered_nogap.fasta",
        query = "hg38"
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
         {params.query}\
         {config[weights]} &&


        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """
