rule calculate_posterior_probabilities:
    input:
        ancestral_probabilities_binary = "{workdir}/{result_dirname}/{query_id}/3_binary_iqtree_ancestral/{query_id}.state.gz",
        ancestral_probabilities_nt = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}.state.gz"
    output:
        ancestral_probabilities_posterior = "{workdir}/{result_dirname}/{query_id}/2_iqtree_ancestral/{query_id}_gapAware_probabilities.tsv",
    log:
        "{workdir}/logs/{result_dirname}/rules/{query_id}_posterior_probabilities.log"
    benchmark:
        "{workdir}/logs/{result_dirname}/benchmarks/{query_id}_posterior_probabilities.out"
    conda:
        "../envs/core.yml"
    resources:
        cpus=4
    shell:
        """
        (
        echo "`date -R`: {rule} started..." &&

        python3 ../scripts/postProba_indelPerc.py\
         --nt_probs {input.ancestral_probabilities_nt} \
         --gap_probs {input.ancestral_probabilities_binary} \
         --binary_out {output.ancestral_probabilities_posterior} &&

        echo "`date -R`: {rule} ended successfully!" ||
        {{ echo "`date -R`: {rule} failed..."; exit 1; }} ) >> {log} 2>&1

        if [ ! -s {output.ancestral_probabilities_posterior} ]; then
            echo "`date -R`: Output is empty. Exiting..." >> {log}
            exit 1
        fi
        """