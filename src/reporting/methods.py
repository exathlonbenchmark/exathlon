"""Module specifying the methods to report and compare.

Each arguments dictionary in `COMPARED_METHODS` corresponds to a method to add to the reporting.
It should be assigned a title at its corresponding index in `COMPARED_METHOD_TITLES`.

For each method, multiple argument values can be provided for a single argument key. In such a case,
a performance for the method will be computed for each arguments combination. The way these multiple
performances by method are further handled depends on the type of report.

For tables, performances are aggregated according to `reporting_args.aggregation_method`.
"""
import os
import copy

import sys
src_path = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(src_path)
# the considered data should be specified in the .env file
from utils.common import USED_DATA


# arguments combinations to compare and corresponding method titles
ARGS = {
    # data arguments
    'data': USED_DATA,

    # spark-specific arguments
    'app_id': 0,
    'trace_types': '.',
    'ignored_anomalies': 'none',

    # datasets constitution arguments
    'n_starting_removed': 0,
    'n_ending_removed': 0,
    'pre_sampling_period': '15s',

    # features alteration and transformation arguments
    'alter_bundles': 'spark_bundles',
    'alter_bundle_idx': 0,
    'data_sampling_period': '15s',
    'data_downsampling_position': 'last',
    'labels_sampling_period': '15s',
    'transform_chain': 'trace_scaling',
    'head_size': 240,
    'online_window_type': 'expanding',
    'regular_pretraining_weight': -1,
    'scaling_method': 'std',
    'reg_scaler_training': 'all.training',
    'minmax_range': [0, 1],
    'pca_n_components': 13,
    'pca_kernel': 'linear',
    'pca_training': 'all.training',
    'fa_n_components': 13,
    'fa_training': 'all.training',

    # normality modeling arguments
    'modeling_n_periods': -1,
    'modeling_data_prop': 1.0,
    'modeling_data_seed': 21,
    'modeling_split': 'stratified.split',
    'modeling_split_seed': 21,
    'n_period_strata': 3,
    'modeling_val_prop': 0.15,
    'modeling_test_prop': 0.15,
    'model_type': 'ae',
    # FORECASTING MODELS #
    'n_back': 40,
    'n_forward': 1,
    # RNN
    'rnn_unit_type': 'lstm',
    'rnn_n_hidden_neurons': [144, 40],
    'rnn_dropout': 0.0,
    'rnn_rec_dropout': 0.0,
    'rnn_optimizer': 'adam',
    'rnn_learning_rate': 7.869 * (10 ** -4),
    'rnn_n_epochs': 200,
    'rnn_batch_size': 32,
    # RECONSTRUCTION MODELS #
    'window_size': 40,
    'window_step': 1,
    # autoencoder
    'ae_latent_dim': 32,
    'ae_type': 'dense',
    'ae_enc_n_hidden_neurons': [200],
    'ae_dec_last_activation': 'linear',
    'ae_dropout': 0.0,
    'ae_dense_layers_activation': 'relu',
    'ae_rec_unit_type': 'lstm',
    'ae_rec_dropout': 0.0,
    'ae_loss': 'mse',
    'ae_optimizer': 'adam',
    'ae_learning_rate': 3.602 * (10 ** -4),
    'ae_n_epochs': 200,
    'ae_batch_size': 32,
    # BiGAN
    'bigan_latent_dim': 32,
    'bigan_enc_type': 'rec',
    'bigan_enc_arch_idx': -1,
    'bigan_enc_rec_n_hidden_neurons': [100],
    'bigan_enc_rec_unit_type': 'lstm',
    'bigan_enc_conv_n_filters': 32,
    'bigan_enc_dropout': 0.0,
    'bigan_enc_rec_dropout': 0.0,
    'bigan_gen_type': 'rec',
    'bigan_gen_last_activation': 'linear',
    'bigan_gen_arch_idx': -1,
    'bigan_gen_rec_n_hidden_neurons': [100],
    'bigan_gen_rec_unit_type': 'lstm',
    'bigan_gen_conv_n_filters': 64,
    'bigan_gen_dropout': 0.0,
    'bigan_gen_rec_dropout': 0.0,
    'bigan_dis_type': 'conv',
    'bigan_dis_arch_idx': 0,
    'bigan_dis_x_rec_n_hidden_neurons': [30, 10],
    'bigan_dis_x_rec_unit_type': 'lstm',
    'bigan_dis_x_conv_n_filters': 32,
    'bigan_dis_x_dropout': 0.0,
    'bigan_dis_x_rec_dropout': 0.0,
    'bigan_dis_z_n_hidden_neurons': [32, 10],
    'bigan_dis_z_dropout': 0.0,
    'bigan_dis_threshold': 0.0,
    'bigan_dis_optimizer': 'adam',
    'bigan_enc_gen_optimizer': 'adam',
    'bigan_dis_learning_rate': 0.0004,
    'bigan_enc_gen_learning_rate': 0.0001,
    'bigan_n_epochs': 200,
    'bigan_batch_size': 32,

    # outlier score assignment arguments
    'scoring_method': 'mse',
    'mse_weight': 0.5,

    # supervised evaluation for assessing scoring performance
    'evaluation_type': 'ad2',
    'recall_alpha': 0.0,
    'recall_omega': 'default',
    'recall_delta': 'flat',
    'recall_gamma': 'dup',
    'precision_omega': 'default',
    'precision_delta': 'flat',
    'precision_gamma': 'dup',
    'f_score_beta': 1.0,

    # outlier score threshold selection arguments
    'thresholding_method': ['std', 'mad', 'iqr'],
    'thresholding_factor': [1.5, 2.0, 2.5, 3.0],
    'n_iterations': [1, 2],
    'removal_factor': [1.0],

    # explanation discovery arguments
    'explanation_method': 'exstream',
    'explained_predictions': 'ground.truth',
    # ED evaluation parameters
    'ed_eval_min_anomaly_length': 1,
    'ed1_consistency_n_disturbances': 5,
    # model-free evaluation
    'mf_eval_min_normal_length': 1,
    'mf_ed1_consistency_sampled_prop': 0.8,
    'mf_ed1_accuracy_n_splits': 5,
    'mf_ed1_accuracy_test_prop': 0.2,
    # model-dependent evaluation
    'md_eval_small_anomalies_expansion': 'before',
    'md_eval_large_anomalies_coverage': 'all',
    # EXstream
    'exstream_fp_scaled_std_threshold': 1.64,
    # MacroBase
    'macrobase_n_bins': 10,
    'macrobase_min_support': 0.4,
    'macrobase_min_risk_ratio': 1.5,
    # LIME
    'lime_n_features': 5,

    # pipeline execution shortcut arguments
    'pipeline_type': 'ad.ed'
}
# uncomment to report the AD results of the paper (after having run the corresponding methods)
COMPARED_ARGS, COMPARED_METHOD_TITLES = [], []
for m, s in zip(['rnn', 'ae', 'bigan'], ['re', 'mse', 'mse.ft']):
    args_dict = copy.deepcopy(ARGS)
    args_dict['model_type'] = m
    args_dict['scoring_method'] = s
    COMPARED_ARGS.append(args_dict)
    COMPARED_METHOD_TITLES.append(m.upper())
# uncomment to report the ED results of the paper (after having run the corresponding methods)
#COMPARED_ARGS, COMPARED_METHOD_TITLES = [], []
#for m in ['macrobase', 'exstream', 'lime']:
#    args_dict = copy.deepcopy(ARGS)
#    args_dict['explanation_method'] = m
#    COMPARED_ARGS.append(args_dict)
#    COMPARED_METHOD_TITLES.append(m.upper())
