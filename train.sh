# Change this part accordingly
# ========================== #
datasource="Virus" # "Virus" "Bacteria" "Tumor"
prob_model="sgld" # "edl", "la", "svdkl", "swag", "vbll", "sgld"
# ========================== #

script=train_"$prob_model".py
max_batch_token=10000 
patience=5
max_train_epochs=50

dataset="$datasource"Binary
pdb_type=ESMFold
seqs=ez_descriptor,esm3_structure_seq,foldseek_seq
seqs_type=full

mutation_rate=""
mutation_prob=""

pooling_head=attention1d

lr=1e-4

for seed in 1 2 3 4 5;
do
    ckpt_root=./ckpt-"$datasource"Immunogen-"$prob_model"-seed-"$seed"

    plm_model_name_list=( "esmc_600m" "Rostlab/ProstT5" "ElnaggarLab/ankh-large" "facebook/esm2_t33_650M_UR50D" "Rostlab/prot_bert" )
    model_name_list=( "esmc_wiln" "prost_t5" "ankh" "esm2" "prot_bert" )

    for ((i=0;i<${#plm_model_name_list[@]};++i)); do
        # Echo info of task to be executed
        plm_model_name="${plm_model_name_list[$i]}"
        model_name="${model_name_list[$i]}"
        echo $model_name

        python $script \
            --plm_model $plm_model_name \
            --num_attention_heads 8 \
            --pooling_method $pooling_head \
            --pooling_dropout 0.1 \
            --dataset_config dataset/$dataset/"$dataset"_"$pdb_type".json \
            --lr $lr \
            --num_workers 1 \
            --seed $seed \
            --gradient_accumulation_steps 1 \
            --max_train_epochs $max_train_epochs \
            --max_batch_token $max_batch_token \
            --patience $patience \
            --structure_seqs $seqs \
            --ckpt_root $ckpt_root \
            --ckpt_dir $model_name \
            --model_name "$pdb_type"_"$model_name"_"$pooling_head"_"$mutation_prob"_"$mutation_rate".pt #&
    done

    # wait
done

echo "DONE"