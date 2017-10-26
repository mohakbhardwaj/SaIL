train_folder = "../../motion_planning_datasets/alternating_gaps/train"
train_oracle_folder = "../SaIL/oracle/saved_oracles/xy/alternating_gaps/train"
valid_folder = "../../motion_planning_datasets/alternating_gaps/validation"
valid_oracle_folder = "../SaIL/oracle/saved_oracles/xy/alternating_gaps/validation"
train_file_start_num = "0"
valid_file_start_num = "800"
model_folder = "../SaIL/learners/trained_models/alternating_gaps"
results_folder = "../SaIL/results/alternating_gaps"


python run_sail_agent.py --train_folder  train_folder --train_oracle_folder  train_oracle_folder --valid_folder  valid_folder --valid_oracle_folder  valid_oracle_folder --train_file_start_num 0 --valid_file_start_num 800 --model_folder model_folder  --results_folder results_folder