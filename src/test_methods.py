import numpy as np
from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import matplotlib.pyplot as plt
import matplotlib
import pickle
from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

import numpy as np
import tflearn
import tensorflow as tf

from stock_trading import StockActor, StockCritic, obs_normalizer, get_model_path, get_result_path, \
                          test_model, get_variable_scope, test_model_multiple

from model.supervised.lstm import StockLSTM
from model.supervised.cnn import StockCNN
from environment.portfolio import PortfolioEnv, MultiActionPortfolioEnv


def visualise_Data():

    with open('utils/datasets/all_eqw','rb') as fr:
        history=pickle.load(fr,encoding='latin1')

    with open('utils/datasets/stock_names','rb') as fr:
        abbreviation=pickle.load(fr,encoding='latin1')


    history = history[:, :, :4]
    num_training_time = history.shape[1]
    num_testing_time = history.shape[1]
    window_length = 3

    # get target history
    target_stocks = ['BLK UN EQUITY','GS UN EQUITY','USB UN EQUITY']
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]
        print(target_history[i])

    # collect testing data
    testing_stocks = ['AMG UN EQUITY', 'BRK/B UN EQUITY', 'MTB UN EQUITY', ]
    testing_history = np.empty(
        shape=(len(testing_stocks), num_testing_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        testing_history[i] = history[abbreviation.index(stock),
                             :num_testing_time, :]

    # dataset for 16 stocks by splitting timestamp
    history, abbreviation = read_stock_history(
        filepath='utils/datasets/stocks_history_target.h5')
    with open('utils/datasets/all_eqw', 'rb') as fr:
        history = pickle.load(fr, encoding='latin1')

    with open('utils/datasets/stock_names', 'rb') as fr:
        abbreviation = pickle.load(fr, encoding='latin1')
    history = history[:, :, :4]

    # 16 stocks are all involved. We choose first 3 years as training data
    num_training_time = 1095
    target_stocks = abbreviation
    target_history = np.empty(
        shape=(len(target_stocks), num_training_time, history.shape[2]))

    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock),
                            :num_training_time, :]
    print((target_history.shape))

    # and last 2 years as testing data.
    testing_stocks = abbreviation
    testing_history = np.empty(
        shape=(len(testing_stocks), history.shape[1] - num_training_time,
               history.shape[2]))
    for i, stock in enumerate(testing_stocks):
        testing_history[i] = history[abbreviation.index(stock),
                             num_training_time:, :]

    print((testing_history.shape))

    nb_classes = len(target_stocks) + 1
    print(target_history.shape)
    print(testing_history.shape)

    if True:
        date_list = [index_to_date(i) for i in range(target_history.shape[1])]
        x = range(target_history.shape[1])
        for i in range(len(target_stocks)):
            plt.figure(i)
            plt.plot(x, target_history[i, :,
                        1])  # open, high, low, close = [0, 1, 2, 3]
            plt.xticks(x[::200], date_list[::200], rotation=30)
            plt.title(target_stocks[i])
            plt.show()


    # common settings
    batch_size = 64
    action_bound = 1.
    tau = 1e-3
    models = []
    model_names = []
    window_length_lst = [3, 7, 14, 21]
    predictor_type_lst = ['cnn', 'lstm']
    use_batch_norm = True


    for window_length in window_length_lst:
        name = 'imit_LSTM%3A window = {}'.format(window_length)
        model_name = 'imitation_lstm_window_{}'.format(window_length)
        model_names.append(model_name)
        # instantiate LSTM model
        lstm_model = StockLSTM(nb_classes, window_length,
                               weights_file='weights/' + name + '.h5')
        lstm_model.build_model(load_weights=True)
        models.append(lstm_model)

        name = 'imit_CNN%3A window = {}'.format(window_length)
        model_name = 'imitation_cnn_window_{}'.format(window_length)
        model_names.append(model_name)
        # instantiate CNN model
        cnn_model = StockCNN(nb_classes, window_length,
                             weights_file='weights/' + name + '.h5')
        cnn_model.build_model(load_weights=True)
        models.append(cnn_model)

    # instantiate environment, 3 stocks, with trading cost, window_length 3, start_date sample each time

    for window_length in window_length_lst:
        for predictor_type in predictor_type_lst:
            name = 'DDPG_window_{}_predictor_{}'.format(window_length,
                                                        predictor_type)
            model_names.append(name)
            tf.reset_default_graph()
            sess = tf.Session()
            tflearn.config.init_training_mode()
            action_dim = [nb_classes]
            state_dim = [nb_classes, window_length]
            variable_scope = get_variable_scope(window_length,
                                                predictor_type,
                                                use_batch_norm)
            with tf.variable_scope(variable_scope):
                actor = StockActor(sess, state_dim, action_dim,
                                   action_bound, 1e-4, tau, batch_size,
                                   predictor_type,
                                   use_batch_norm)
                critic = StockCritic(sess=sess, state_dim=state_dim,
                                     action_dim=action_dim, tau=1e-3,
                                     learning_rate=1e-3,
                                     num_actor_vars=actor.get_num_trainable_vars(),
                                     predictor_type=predictor_type,
                                     use_batch_norm=use_batch_norm)
                actor_noise = OrnsteinUhlenbeckActionNoise(
                    mu=np.zeros(action_dim))

                model_save_path = get_model_path(window_length,
                                                 predictor_type,
                                                 use_batch_norm)
                summary_path = get_result_path(window_length,
                                               predictor_type,
                                               use_batch_norm)

                ddpg_model = DDPG(None, sess, actor, critic, actor_noise,
                                  obs_normalizer=obs_normalizer,
                                  config_file='config/stock.json',
                                  model_save_path=model_save_path,
                                  summary_path=summary_path)
                ddpg_model.initialize(load_weights=True, verbose=False)
                models.append(ddpg_model)

    env = MultiActionPortfolioEnv(target_history, target_stocks,
                                  model_names[8:], steps=500,
                                  sample_start_date='2012-10-30')

    test_model_multiple(env, models[8:])





def main():

    visualise_Data()


if __name__ == '__main__':
    main()