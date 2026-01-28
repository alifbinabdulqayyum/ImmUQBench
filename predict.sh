# Change this part accordingly
# ========================== #
prob_model="sgld" #"dvbll", "mcd", "la", "svdkl", "swag", "ts", "edl"
datasource="Virus" # "Virus" "Bacteria" "Tumor"
targetsource="Virus" # "Virus" "Bacteria" "Tumor"
# ========================== #

script=predict_"$prob_model".py
testset="test"

for seed in 1 2 3 4 5;
do
    python ./predict_files/$script \
        --datasource $datasource \
        --targetsource $targetsource \
        --testset $testset \
        --num_runs 64 \
        --seed $seed
done

echo "DONE"