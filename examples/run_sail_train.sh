#!/bin/bash

TRAIN_FOLDER="../../motion_planning_datasets/shifting_gaps/train"
TRAIN_ORACLE_FOLDER="../SaIL/oracle/saved_oracles/xy/shifting_gaps/train"
VALIDATION_FOLDER="../../motion_planning_datasets/shifting_gaps/validation"
VALIDATION_ORACLE_FOLDER="../SaIL/oracle/saved_oracles/xy/shifting_gaps/validation"
MODEL_FOLDER="../SaIL/learners/trained_models/xy/shifting_gaps"
TRAIN_FILE_START_NUM="0"
VALIDATION_FILE_START_NUM="800"
PRETRAINED_MODEL="../SaIL/learners/trained_models/xy/shifting_gaps/200_70"
ORACLE_FILE_TYPE="json"

gdb -ex r --args python sail_xy_train.py --train_folder  ${TRAIN_FOLDER} --train_oracle_folder  ${TRAIN_ORACLE_FOLDER} --validation_folder  ${VALIDATION_FOLDER} --validation_oracle_folder  ${VALIDATION_ORACLE_FOLDER} --model_folder ${MODEL_FOLDER} --pretrained_model ${PRETRAINED_MODEL} --train_file_start_num ${TRAIN_FILE_START_NUM} --validation_file_start_num ${VALIDATION_FILE_START_NUM} --oracle_file_type ${ORACLE_FILE_TYPE}
# python sail_xy_train.py --train_folder  ${TRAIN_FOLDER} --train_oracle_folder  ${TRAIN_ORACLE_FOLDER} --validation_folder  ${VALIDATION_FOLDER} --validation_oracle_folder  ${VALIDATION_ORACLE_FOLDER} --model_folder ${MODEL_FOLDER} --pretrained_model ${PRETRAINED_MODEL} --train_file_start_num ${TRAIN_FILE_START_NUM} --validation_file_start_num ${VALIDATION_FILE_START_NUM} --oracle_file_type ${ORACLE_FILE_TYPE}
