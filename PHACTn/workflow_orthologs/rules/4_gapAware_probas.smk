
rule calculate_posterior_probabilities:
    input:
        ancestral_probabilities_binary = "{workdir}/{score_result_dirname}/{query_id}/3_binary_raxmlng_ancestral/{query_id}.raxml.ancestralProbs",
        ancestral_probabilities_nt = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}.raxml.ancestralProbs",
    output:
        ancestral_probabilities_posterior = "{workdir}/{score_result_dirname}/{query_id}/2_raxmlng_ancestral/{query_id}_gapAware_probabilities.tsv",
    log:
        "{workdir}/logs/{score_result_dirname}/rules/{query_id}_posterior_probabilities.log"
    benchmark:
        "{workdir}/logs/{score_result_dirname}/benchmarks/{query_id}_posterior_probabilities.out"
    conda:
        "../conda_env.yml"
    resources:
        cpus=2
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
