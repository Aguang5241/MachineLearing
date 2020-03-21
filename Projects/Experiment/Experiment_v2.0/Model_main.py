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
    testing_data_file_path = 'Projects/Experiment/Experiment_v2.0/res/Data/ProAveTestingDataFiltered.csv'
    features = 3

    ############################################################
    #                        Train                             #
    ############################################################

    train = input('Train or Not(Y/N)')
    if train.lower() == 'y':
        learning_rate = 3e-4
        loss_threashold_value = 1e-1
        error = e.repeat(6, 1)
        loop_max = 1e5
        index = np.random.randn(1)
        path = 'Projects/Experiment/Experiment_v2.0/res/Train/ProAve/%.3f/' % index
        training_data_file_path = 'Projects/Experiment/Experiment_v2.0/res/Data/ProAveTrainingData.csv'
        upper_limit = 3
        lower_limit = -(upper_limit / 8) * 2
        train_parameters_list = [learning_rate, loss_threashold_value, error, loop_max,
                                 path, training_data_file_path, testing_data_file_path,
                                 upper_limit, lower_limit, Net, features]
        training_break = Model_train.main(train_parameters_list)
        if not training_break:
            print('\n+++++++++++++++++++Model train complete+++++++++++++++++\n')
            print('Train: the index of file ---> %.3f' % index)

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
        re_path = 'Projects/Experiment/Experiment_v2.0/res/Retrain/%.3f/' % re_index
        re_training_data_file_path = 'Projects/Experiment/Experiment_v2.0/res/Data/ProAveTrainingData.csv'
        # re_training_data_file_path = 'Projects/Experiment/Experiment_v2.0/res/Data/TrainingDataPlus1.csv'
        # re_training_data_file_path = 'Projects/Experiment/Experiment_v2.0/res/Data/TrainingDataPlus2.csv'
        re_upper_limit = 1e-1
        re_lower_limit = -(re_upper_limit / 8) * 2
        old_model_path = 'Projects/Experiment/Experiment_v2.0/res/Train/ProAve/-2.449/model.pkl'
        retrain_parameters_list = [re_learning_rate, re_loss_threashold_value, re_error, re_loop_max,
                                   re_path, re_training_data_file_path, testing_data_file_path, re_upper_limit, re_lower_limit, Net, features, old_model_path]
        retraining_break = Model_retrain.main(retrain_parameters_list)
        if not retraining_break:
            print('\n+++++++++++++++++Model retrain complete+++++++++++++++++\n')
            print('Retrain: the index of file ---> %.3f' % re_index)

    ############################################################
    #                       Process                            #
    ############################################################

    process = input('Process or Not(Y/N)')
    if process.lower() == 'y':
        pro_path = 'Projects/Experiment/Experiment_v2.0/res/Process/'
        # pro_training_data_file_path = 'Projects/Experiment/Experiment_v1.0/res/Data/TrainingData.csv'
        # pro_training_data_file_path = 'Projects/Experiment/Experiment_v1.0/res/Data/TrainingDataPlus1.csv'
        pro_training_data_file_path = 'Projects/Experiment/Experiment_v2.0/res/Data/ProAveTrainingData.csv'
        model_path = 'Projects/Experiment/Experiment_v2.0/res/Retrain/-0.338/model.pkl'
        pro_parameters_list = [pro_path, pro_training_data_file_path,
                               testing_data_file_path, model_path]
        Model_progress.main(pro_parameters_list)
        print('\n++++++++++++++++Model Process complete++++++++++++++++++')


if __name__ == '__main__':
    main()
