###########################################################
#                net shape: n-12-6-3                      #
#                layer function: Linear                   #
#                standard: True                           #
#                activation function: ReLU                #
#                loss function: MSELoss                   #
#                optimizer: Adam                          #
###########################################################

import Model_train
import Model_retrain
import Model_process
import Model_NN

import torch
import numpy as np


# 定义神经网络
Net = Model_NN.Net


def main():

    e = torch.tensor([3, 3, 0.1]).float()
    predicting_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\Data\\Perdicting_data.csv'
    training_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\Data\\Training_data.csv'
    features = 4
    layer_1 = 12
    layer_2 = 6

    ############################################################
    #                        Train                             #
    ############################################################

    train = input('Train or Not(Y/N)')
    if train.lower() == 'y':
        learning_rate = 1e-4
        loss_threashold_value = 0.892
        loop_max = 1e5
        index = np.random.randn(1)
        path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Train\\%.3f\\' % index
        train_start_index = 0
        train_end_index = 11
        error = e.repeat(train_end_index - train_start_index, 1)
        upper_limit = 3
        lower_limit = -(upper_limit / 8) * 2
        train_parameters_list = [learning_rate, loss_threashold_value, error, loop_max, path,
                                 train_start_index, train_end_index, training_data_file_path, predicting_data_file_path,
                                 upper_limit, lower_limit, Net, features, layer_1, layer_2]
        training_break = Model_train.main(train_parameters_list)
        if not training_break:
            print('\n+++++++++++++++++++Model train complete+++++++++++++++++')
            print('Train: the index of file ---> %.3f\n' % index)

    ############################################################
    #                       Retrain                            #
    ############################################################

    retrain = input('Retrain or Not(Y/N)')
    if retrain.lower() == 'y':
        re_learning_rate = 1e-4
        re_loss_threashold_value = 1e-2
        re_loop_max = 1e5
        re_index = np.random.randn(1)
        re_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Retrain\\%.3f\\' % re_index
        train_start_index = 0
        train_end_index = 11
        re_error = e.repeat(train_end_index - train_start_index, 1)
        re_upper_limit = 3
        re_lower_limit = -(re_upper_limit / 8) * 2
        old_model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Retrain\\0-8_1e-2\\model.pkl'
        retrain_parameters_list = [re_learning_rate, re_loss_threashold_value, re_error, re_loop_max, re_path,
                                   train_start_index, train_end_index, training_data_file_path, predicting_data_file_path, re_upper_limit, re_lower_limit, Net, features, layer_1, layer_2, old_model_path]
        retraining_break = Model_retrain.main(retrain_parameters_list)
        if not retraining_break:
            print('\n+++++++++++++++++Model retrain complete+++++++++++++++++')
            print('Retrain: the index of file ---> %.3f\n' % re_index)

    ############################################################
    #                       Process                            #
    ############################################################

    process = input('Process or Not(Y/N)')
    if process.lower() == 'y':
        pro_index = np.random.randn(1)
        pro_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Process\\%.3f\\' % pro_index
        train_start_index = 0
        train_end_index = 11
        model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Retrain\\0-4-8-11_1e-2\\model.pkl'
        pro_parameters_list = [pro_path, train_start_index, train_end_index,
                               training_data_file_path, predicting_data_file_path,
                               model_path]
        Model_process.main(pro_parameters_list)
        print('\n++++++++++++++++Model Process complete++++++++++++++++++')
        print('Progress: the index of file ---> %.3f' % pro_index)


if __name__ == '__main__':
    main()
