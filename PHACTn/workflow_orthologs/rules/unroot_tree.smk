import os 
rule unroot_fasttree:
    input:
        treefile = lambda wildcards: f"{config['MLtree_file_pattern']}".format(query_id=wildcards.query_id),
    output:
        treefile_unrooted = "{workdir}/{score_result_dirname}/{query_id}/2_unrooted_tree/{query_id}_precursor_annotated.treefile_unrooted",
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}/unroot.out"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}/unroot.out"
    conda:
        "../../conda_env.yml"
    shell:
        """
        (
         echo "`date -R`: {rule} started..." &&


        Rscript scripts/unroot_tree.R \
         {input.treefile} {output.treefile_unrooted} &&

        
        echo "`date -R`: {rule} ended successfully!" ||
         {{ echo "`date -R`: {rule} failed..."; exit 1; }}  )  >> {log} 2>&1

        if [ ! -s {output.treefile_unrooted} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """