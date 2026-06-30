#!/bin/bash

# # --- ORNSTEIN UHLENBECK ---
# declare -A T_values=( ["tiny_T"]=0.000025 ["small_T"]=0.00025 ["med_T"]=0.0025)
# T_order=("tiny_T" "small_T" "med_T")
# sigmas=(1 3 6)

# for T_name in "${T_order[@]}"; do
#     T="${T_values[$T_name]}"
#     for sigma in "${sigmas[@]}"; do
#         poetry run python scripts/generate_signal.py \
#             --noise_name ornstein_uhlenbeck \
#             --out_path "./data/ou/N_8192/${T_name}/${sigma}_sigma" \
#             --extra_args 8192 0.5 1 "$sigma" "$T"
#     done
# done
# --- Stationary OU ---
# poetry run python scripts/generate_signal.py \
#             --noise_name stationary_ornstein_uhlenbeck \
#             --out_path "./data/stat_ou/N_8192/output" \
#             --extra_args 8192 0.5 1 0.025

# poetry run python scripts/generate_signal.py \
#             --noise_name stationary_ornstein_uhlenbeck
#             --out_path "./data/stat_ou/N_65536/output" \
#             --extra_args 65536 0.5 1 0.025

# --- Stat TLF --
poetry run python scripts/generate_signal.py \
            --noise_name one_over_f \
            --out_path "./data/stat_tlf_bath/N_65536/output" \
            --extra_args 65536 200 0.001 1000 1 0 0.02


# --- TLF BATH WITH LOG-UNIFORM GAMMAS ---

# declare -A T_values=( ["alphaplus125"]=0.00135672 ["alphaplus150"]=0.00241263 ["alphaplus175"]=0.00429032)
# T_order=("alphaplus125" "alphaplus150" "alphaplus175")
# sigmas=(1 3 6)

# for T_name in "${T_order[@]}"; do
#     T="${T_values[$T_name]}"
#     for sigma in "${sigmas[@]}"; do
#         poetry run python scripts/generate_signal.py \
#             --noise_name one_over_f \
#             --out_path "./data/tlf_bath_narrow/N_65536/${T_name}/${sigma}_sigma" \
#             --extra_args 65536 200 0.001 0.1 1 "$sigma" "$T"
#     done
# done