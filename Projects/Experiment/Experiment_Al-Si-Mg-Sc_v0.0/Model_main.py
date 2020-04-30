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

    # Common
    training_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Data\\Training_data_CT.csv'
    predicting_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Regenerate\\6data\\generate_results.csv'
    # # predicting_data_file_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Data\\Predicting_data_CT.csv'
    features = 3
    index = np.random.randn(1)
    loop_max = 1e5
    train_start_index = 0

    # Model-I
    e = torch.tensor([3, 3, 0.1]).float()
    ANN_I_layer_1 = 12
    ANN_I_layer_2 = 6

    # Model-II
    EL_Sc_predict = torch.from_numpy(
        np.transpose([np.linspace(0, 0.75, 751)])).float()
    ANN_II_layer_1 = 8

    ############################################################
    #                      Generate                            #
    ############################################################

    generate = input('Generate or Not(Y/N)')
    if generate.lower() == 'y':
        # 定义神经网络
        Net = Model_ANN_II.Net

        path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Generate\\%.3f\\' % index
        learning_rate = 1e-3
        loss_threashold_value = 1e-9
        train_end_index = 5
        parameters_list = [training_data_file_path, features, loop_max, EL_Sc_predict,
                           ANN_II_layer_1, Net, path, learning_rate, loss_threashold_value,
                           train_start_index, train_end_index]
        training_break = Model_generate.main(parameters_list)
        if not training_break:
            print('\n+++++++++++++++++Model generate complete++++++++++++++++')
            print('Generate: the index of file ---> %.3f\n' % index)

    ############################################################
    #                      Regenerate                          #
    ############################################################

    regenerate = input('Regenerate or Not(Y/N)')
    if regenerate.lower() == 'y':
        # 定义神经网络
        Net = Model_ANN_II.Net

        path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Regenerate\\%.3f\\' % index
        old_model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Regenerate\\5data\\generate_model.pkl'
        learning_rate = 1e-3
        loss_threashold_value = 5e-11
        train_end_index = 6
        add = True
        error = np.array([[[3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3]],
                          [[3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3]],
                          [[2, 2, 2, 2, 2],
                           [2, 2, 2, 2, 2]],
                          [[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]]])
        parameters_list = [training_data_file_path, features, loop_max, EL_Sc_predict,
                           ANN_II_layer_1, Net, path, old_model_path, learning_rate,
                           loss_threashold_value, train_start_index, train_end_index,
                           error, add]
        training_break = Model_regenerate.main(parameters_list)
        if not training_break:
            print('\n++++++++++++++++Model regenerate complete+++++++++++++++')
            print('Regenerate: the index of file ---> %.3f\n' % index)

    ############################################################
    #                        Train                             #
    ############################################################

    train = input('Train or Not(Y/N)')
    if train.lower() == 'y':
        # 定义神经网络
        Net = Model_ANN_I.Net

        path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Train\\%.3f\\' % index
        learning_rate = 1e-3
        loss_threashold_value = 1e-2
        train_end_index = 6
        error = e.repeat(train_end_index - train_start_index, 1)
        upper_limit = 3
        lower_limit = -(upper_limit / 8) * 2
        parameters_list = [training_data_file_path, predicting_data_file_path, features,
                           loop_max, ANN_I_layer_1, ANN_I_layer_2, Net, path, learning_rate,
                           loss_threashold_value, train_start_index, train_end_index, error,
                           upper_limit, lower_limit]
        training_break = Model_train.main(parameters_list)
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

        path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Retrain\\%.3f\\' % index
        old_model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Train\\6data\\model.pkl'
        learning_rate = 1e-3
        loss_threashold_value = 1e-2
        train_end_index = 6
        error = e.repeat(train_end_index - train_start_index, 1)
        upper_limit = 3
        lower_limit = -(upper_limit / 8) * 2
        parameters_list = [training_data_file_path, predicting_data_file_path, features,
                           loop_max, ANN_I_layer_1, ANN_I_layer_2, Net, path, old_model_path,
                           learning_rate, loss_threashold_value, train_start_index,
                           train_end_index, error, upper_limit, lower_limit]
        training_break = Model_retrain.main(parameters_list)
        if not training_break:
            print('\n+++++++++++++++++Model retrain complete+++++++++++++++++')
            print('Retrain: the index of file ---> %.3f\n' % index)

    ############################################################
    #                       Process                            #
    ############################################################

    process = input('Process or Not(Y/N)')
    if process.lower() == 'y':
        # 定义神经网络
        Net = Model_ANN_I.Net

        path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Process\\%.3f\\' % index
        model_path = r'Projects\\Experiment\\Experiment_Al-Si-Mg-Sc_v0.0\\res\\Retrain\\5data\\model.pkl'
        train_end_index = 5
        add = False
        error = np.array([[[3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3]],
                          [[3, 3, 3, 3, 3],
                           [3, 3, 3, 3, 3]],
                          [[1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1]]])
        parameters_list = [training_data_file_path, predicting_data_file_path,
                           path, model_path, train_start_index,
                           train_end_index, error, add]
        Model_process.main(parameters_list)
        print('\n++++++++++++++++Model Process complete++++++++++++++++++')
        print('Progress: the index of file ---> %.3f' % index)


if __name__ == '__main__':
    main()
