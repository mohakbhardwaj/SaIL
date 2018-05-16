#!/bin/bash

TEST_FOLDER_1="../../motion_planning_datasets/alternating_gaps/validation/"
TEST_FOLDER_2="../../motion_planning_datasets/bugtrap_forest/validation/"
TEST_FOLDER_3="../../motion_planning_datasets/forest/validation/"
TEST_FOLDER_4="../../motion_planning_datasets/gaps_and_forest/validation/"
TEST_FOLDER_5="../../motion_planning_datasets/mazes/validation/"
TEST_FOLDER_6="../../motion_planning_datasets/multiple_bugtraps/validation/"
TEST_FOLDER_7="../../motion_planning_datasets/shifting_gaps/validation/"
TEST_FOLDER_8="../../motion_planning_datasets/single_bugtrap/validation/"

ORACLE_FOLDER_1="../SaIL/oracle/saved_oracles/xy/alternating_gaps/validation/"
ORACLE_FOLDER_2="../SaIL/oracle/saved_oracles/xy/bugtrap_forest/validation/"
ORACLE_FOLDER_3="../SaIL/oracle/saved_oracles/xy/forest/validation/"
ORACLE_FOLDER_4="../SaIL/oracle/saved_oracles/xy/gaps_and_forest/validation/"
ORACLE_FOLDER_5="../SaIL/oracle/saved_oracles/xy/mazes/validation/"
ORACLE_FOLDER_6="../SaIL/oracle/saved_oracles/xy/multiple_bugtraps/validation/"
ORACLE_FOLDER_7="../SaIL/oracle/saved_oracles/xy/shifting_gaps/validation/"
ORACLE_FOLDER_8="../SaIL/oracle/saved_oracles/xy/single_bugtrap/validation/"

MODEL_FILE_1="../SaIL/learners/trained_models/xy/alternating_gaps/200_70" #iter_7_features_17_num_train_envs_200_num_valid_envs_70"
MODEL_FILE_2="../SaIL/learners/trained_models/xy/bugtrap_forest/"
MODEL_FILE_3="../SaIL/learners/trained_models/xy/forest/200_50"
MODEL_FILE_4="../SaIL/learners/trained_models/xy/gaps_and_forest/iter_5_features_17_num_train_envs_200_num_valid_envs_70"
MODEL_FILE_5="../SaIL/learners/trained_models/xy/mazes/200_70"
MODEL_FILE_6="../SaIL/learners/trained_models/xy/multiple_bugtraps/train_iter_5_features_17_num_train_envs_200_num_valid_envs_70"
MODEL_FILE_7="../SaIL/learners/trained_models/xy/shifting_gaps/train_iter_5_features_17_num_train_envs_200_num_valid_envs_70"
MODEL_FILE_8="../SaIL/learners/trained_models/xy/single_bugtrap/train_iter_5_features_17_num_train_envs_200_num_valid_envs_70"

RESULTS_FOLDER_1="../SaIL/results/xy/alternating_gaps/"
RESULTS_FOLDER_2="../SaIL/results/xy/bugtrap_forest/"
RESULTS_FOLDER_3="../SaIL/results/xy/forest/"
RESULTS_FOLDER_4="../SaIL/results/xy/gaps_and_forest/"
RESULTS_FOLDER_5="../SaIL/results/xy/mazes/"
RESULTS_FOLDER_6="../SaIL/results/xy/multiple_bugtraps/"
RESULTS_FOLDER_7="../SaIL/results/xy/shifting_gaps/"
RESULTS_FOLDER_8="../SaIL/results/xy/single_bugtrap/"

NUM_ENVS=10
TEST_FILE_START_NUM=800
ORACLE_FILE_TYPE="json"



python sail_xy_test.py --test_folders ${TEST_FOLDER_5} --test_oracle_folders ${ORACLE_FOLDER_5} --model_files ${MODEL_FILE_5} --result_folders ${RESULTS_FOLDER_5} --num_envs ${NUM_ENVS} --test_file_start_num ${TEST_FILE_START_NUM} --oracle_file_type ${ORACLE_FILE_TYPE} # --visualize 

