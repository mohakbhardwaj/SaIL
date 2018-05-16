#!/bin/bash

TRAIN_FOLDER="../../motion_planning_datasets/alternating_gaps/train"
TRAIN_ORACLE_FOLDER="../SaIL/oracle/saved_oracles/xy/alternating_gaps/train"
VALIDATION_FOLDER="../../motion_planning_datasets/alternating_gaps/validation"
VALIDATION_ORACLE_FOLDER="../SaIL/oracle/saved_oracles/xy/alternating_gaps/validation"
MODEL_FOLDER="../SaIL/learners/trained_models/xy/alternating_gaps"
TRAIN_FILE_START_NUM="0"
VALIDATION_FILE_START_NUM="800"
#PRETRAINED_MODEL=" "
RESULTS_FOLDER="../SaIL/results/xy/alternating_gaps"
ORACLE_FILE_TYPE="json"

python sail_xy_train.py --train_folder  ${TRAIN_FOLDER} --train_oracle_folder  ${TRAIN_ORACLE_FOLDER} --validation_folder  ${VALIDATION_FOLDER} --validation_oracle_folder  ${VALIDATION_ORACLE_FOLDER} --model_folder ${MODEL_FOLDER} --results_folder ${RESULTS_FOLDER} --train_file_start_num ${TRAIN_FILE_START_NUM} --validation_file_start_num ${VALIDATION_FILE_START_NUM} --oracle_file_type ${ORACLE_FILE_TYPE}
