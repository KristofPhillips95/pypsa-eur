#!/bin/bash

for resolution in 37 50 64 96 128 292 256
do echo ${resolution}
snakemake -c1 results/plots/elec_s_${resolution}_a_ec_lcopt_Co2L-24H_p_nom.pdf
snakemake -c1 results/plots/elec_s_${resolution}_c_ec_lcopt_Co2L-24H_p_nom.pdf

snakemake -c1 results/summaries/elec_s_${resolution}_a_ec_lcopt_Co2L-24H_all
snakemake -c1 results/summaries/elec_s_${resolution}_c_ec_lcopt_Co2L-24H_all

done