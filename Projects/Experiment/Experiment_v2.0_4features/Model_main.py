###########################################################
#                net shape: n-10-5-3                      #
#                layer function: Linear                   #
#                standard: True                           #
#                activation function: ReLU                #
#                loss function: MSELoss                   #
#                optimizer: Adam                          #
###########################################################

import Model_train
import Model_retrain
import Model_progress
import Model_NN

import torch
import numpy as np


# 定义神经网络
Net = Model_NN.Net


def main():

    e = torch.tensor([3, 3, 0.1]).float()
    testing_data_file_path = 'Projects/Experiment/Experiment_v2.0_4features/res/Data/TestingDataFiltered.csv'
    training_data_file_path = 'Projects/Experiment/Experiment_v2.0_4features/res/Data/TrainingData.csv'
    features = 4

    ############################################################
    #                        Train                             #
    ############################################################

    train = input('Train or Not(Y/N)')
    if train.lower() == 'y':
        learning_rate = 1e-4
        loss_threashold_value = 1e-2
        error = e.repeat(6, 1)
        loop_max = 1e5
        index = np.random.randn(1)
        path = 'Projects/Experiment/Experiment_v2.0_4features/res/Train/%.3f/' % index
        upper_limit = 3
        lower_limit = -(upper_limit / 8) * 2
        train_parameters_list = [learning_rate, loss_threashold_value, error, loop_max,
                                 path, training_data_file_path, testing_data_file_path,
                                 upper_limit, lower_limit, Net, features]
        training_break = Model_train.main(train_parameters_list)
        if not training_break:
            print('\n+++++++++++++++++++Model train complete+++++++++++++++++')
            print('Train: the index of file ---> %.3f\n' % index)

    ############################################################
    #                       Retrain                            #
    ############################################################

    retrain = input('Retrain or Not(Y/N)')
    if retrain.lower() == 'y':
        re_learning_rate = 5e-5
        re_loss_threashold_value = 1e-2
        re_error = e.repeat(6, 1)
        re_loop_max = 1e5
        re_index = np.random.randn(1)
        re_path = 'Projects/Experiment/Experiment_v2.0_4features/res/Retrain/%.3f/' % re_index
        re_upper_limit = 1e-1
        re_lower_limit = -(re_upper_limit / 8) * 2
        old_model_path = 'Projects/Experiment/Experiment_v2.0_4features/res/Train/-2.449/model.pkl'
        retrain_parameters_list = [re_learning_rate, re_loss_threashold_value, re_error, re_loop_max,
                                   re_path, training_data_file_path, testing_data_file_path, re_upper_limit, re_lower_limit, Net, features, old_model_path]
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
        pro_path = 'Projects/Experiment/Experiment_v2.0_4features/res/Process/%.3f/' % pro_index
        model_path = 'Projects/Experiment/Experiment_v2.0_4features/res/Retrain/-0.338/model.pkl'
        pro_parameters_list = [pro_path, training_data_file_path,
                               testing_data_file_path, model_path]
        Model_progress.main(pro_parameters_list)
        print('\n++++++++++++++++Model Process complete++++++++++++++++++')
        print('Progress: the index of file ---> %.3f' % pro_index)


if __name__ == '__main__':
    main()
