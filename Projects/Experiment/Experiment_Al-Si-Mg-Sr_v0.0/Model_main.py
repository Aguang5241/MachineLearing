###########################################################
#                net shape: n-12-6-3                      #
#                layer function: Linear                   #
#                standard: True                           #
#                activation function: ReLU                #
#                loss function: MSELoss                   #
#                optimizer: Adam                          #
###########################################################

import Model_generate
import Model_regenerate
import Model_train
import Model_retrain
import Model_process
import Model_ANN_I
import Model_ANN_II

import torch
import numpy as np


def main():

    # Model-I
    e = torch.tensor([3, 3, 0.1]).float()
    # predicting_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Regenerate\\1-8-4_v2_8data\\generate_results.csv'
    predicting_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Data\\Predicting_data.csv'
    training_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Data\\Training_data_part.csv'
    features = 4
    layer_1 = 12
    layer_2 = 6

    # Model-II
    EL_Sr_predict = torch.from_numpy(
        np.transpose([np.linspace(0, 0.125, 251)])).float()
    hidden_layer = 8

    ############################################################
    #                      Generate                            #
    ############################################################

    generate = input('Generate or Not(Y/N)')
    if generate.lower() == 'y':
        # 定义神经网络
        Net = Model_ANN_II.Net

        ge_index = np.random.randn(1)
        ge_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Generate\\%.3f\\' % ge_index
        ge_learning_rate = 1e-3
        ge_loop_max = 1e5
        ge_loss_threashold_value = 1e-6
        train_start_index = 0
        train_end_index = 6
        generate_parameters_list = [training_data_file_path, ge_path, features, hidden_layer,
                                    ge_learning_rate, ge_loop_max, ge_loss_threashold_value,
                                    EL_Sr_predict, Net, train_start_index, train_end_index]
        ge_training_break = Model_generate.main(generate_parameters_list)
        if not ge_training_break:
            print('\n+++++++++++++++++Model generate complete++++++++++++++++')
            print('Generate: the index of file ---> %.3f\n' % ge_index)

    ############################################################
    #                      Regenerate                          #
    ############################################################

    regenerate = input('Regenerate or Not(Y/N)')
    if regenerate.lower() == 'y':
        # 定义神经网络
        Net = Model_ANN_II.Net

        rege_index = np.random.randn(1)
        rege_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Regenerate\\%.3f\\' % rege_index
        rege_learning_rate = 1e-4
        rege_loop_max = 1e5
        rege_loss_threashold_value = 3.11e-7
        rege_old_model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Generate\\1-8-4_v2\\generate_model.pkl'
        train_start_index = 0
        train_end_index = 8
        regenerate_parameters_list = [training_data_file_path, rege_path, features, hidden_layer,
                                      rege_learning_rate, rege_loop_max, rege_loss_threashold_value,
                                      EL_Sr_predict, Net, rege_old_model_path, train_start_index, train_end_index]
        rege_training_break = Model_regenerate.main(regenerate_parameters_list)
        if not rege_training_break:
            print('\n++++++++++++++++Model regenerate complete+++++++++++++++')
            print('Regenerate: the index of file ---> %.3f\n' % rege_index)

    ############################################################
    #                        Train                             #
    ############################################################

    train = input('Train or Not(Y/N)')
    if train.lower() == 'y':
        # 定义神经网络
        Net = Model_ANN_I.Net

        learning_rate = 1e-4
        loss_threashold_value = 1e-2
        loop_max = 1e5
        index = np.random.randn(1)
        path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Train\\%.3f\\' % index
        train_start_index = 0
        train_end_index = 6
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
        # 定义神经网络
        Net = Model_ANN_I.Net

        re_learning_rate = 1e-4
        re_loss_threashold_value = 1e-2
        re_loop_max = 1e5
        re_index = np.random.randn(1)
        re_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Retrain\\%.3f\\' % re_index
        train_start_index = 0
        train_end_index = 8
        re_error = e.repeat(train_end_index - train_start_index, 1)
        re_upper_limit = 3
        re_lower_limit = -(re_upper_limit / 8) * 2
        old_model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Train\\1-8-4_v2\\model.pkl'
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
        # 定义神经网络
        Net = Model_ANN_I.Net

        pro_index = np.random.randn(1)
        pro_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Process\\%.3f\\' % pro_index
        train_start_index = 0
        train_end_index = 8
        model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sr_v0.0\\res\\Retrain\\1-8-4_v2_8data\\model.pkl'
        pro_parameters_list = [pro_path, train_start_index, train_end_index,
                               training_data_file_path, predicting_data_file_path,
                               model_path]
        Model_process.main(pro_parameters_list)
        print('\n++++++++++++++++Model Process complete++++++++++++++++++')
        print('Progress: the index of file ---> %.3f' % pro_index)


if __name__ == '__main__':
    main()
