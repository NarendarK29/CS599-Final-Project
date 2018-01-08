import heuristrats
import datetime as dt
import pass_db
import pytz
import pandas as pd
from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

import numpy as np
import tflearn
import tensorflow as tf

from stock_trading import StockActor, StockCritic, obs_normalizer, \
    get_model_path, get_result_path, \
    test_model, get_variable_scope, test_model_multiple

from model.supervised.lstm import StockLSTM
from model.supervised.cnn import StockCNN
from sortedcontainers import SortedDict





class Strategy_all_stocks:

    def __init__(self,tickers):
        self.recommendation=None
        self.tickers=tickers
        self.all_model_agents = self.init_model()

    def initialize(self, strat):
        pass

    def init_model(self):
        # common settings
        batch_size = 64
        action_bound = 1.
        tau = 1e-3

        models = []
        model_names = []
        window_length_lst = [3, 7, 14, 21]
        predictor_type_lst = ['cnn' ,'lstm']
        use_batch_norm = True

        nb_classes=17

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

        print("model names",model_names)

        return models


    def convert_to_state(self,df):
        columns = ['NU_CLOSE', 'NU_HIGH', 'NU_OPEN', 'NU_LOW', 'NU_PX_VOLUME']
        t=self.tickers

        df.dropna(inplace=True)

        #print(df)
        #print(len(df))

        final_df = np.zeros((len(t)+1, len(df), len(columns)))

        #print(final_df.shape)

        final_df[0] = np.ones((len(df),len(columns)))
        i = 1
        for name in t:
            df_organ = df[name][
                ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
            ent = np.array(df_organ.as_matrix())
            #print(ent.shape)
            final_df[i] = ent
            i += 1

        #print(final_df)
        #print(final_df.shape)

        return final_df

    def get_recommendation(self, data, context):

        print "GET recomendation"

        df  =  data["market"].copy()
        #print df

        last_all_price=[]
        for name in self.tickers:
            df_last = df.iloc[-1,df.columns.get_level_values(0) == name ]

            #print(df_last)

            last_all_price.append(df_last[name]['CLOSE'])

        print ( "Last all",last_all_price )

        observation = self.convert_to_state(df)

        observation = observation[:, :, :4]

        # observation, info = env.reset()
        # done = False
        # while not done:

        action = self.all_model_agents[8].predict_single(observation)

        #     observation, _, done, _ = env.step(action)
        # env.render()

        print("action is",action)
        print("sum of action is" ,sum(action))



        #last_price = data['prices'][self.ticker].iloc[-1]
        amount = context['amount']
        #n_val = int(amount/last_price)

        #action is % of how much we want to invest in each one of them
        #excluding first one!!!
        n_val_all={}
        for i in  range(len(self.tickers)):
            amt_port = amount * action[i+1]
            n_val = int(amt_port/last_all_price[i])
            n_val_all[self.tickers[i]]=dict([('n_shares',n_val)])






        print "Avabuying n shares : ",n_val_all


        block = {
            'orders': n_val_all,
            'type': 'L',
            'score': 1.0,
            'id': 'All_Stocks',
            'order_type': 'MKT'
        }

        L=1


        return {'blocks': [block], 'target_blocks_L':L}


def test_strategy():


    tickers = ['BLK UN EQUITY', 'GS UN EQUITY', 'USB UN EQUITY', 'AMG UN EQUITY',
         'BRK/B UN EQUITY', 'MTB UN EQUITY',
         'AMP UN EQUITY', 'BBT UN EQUITY', 'WLTW UW EQUITY', 'CMA UN EQUITY',
         'CB UN EQUITY', 'MS UN EQUITY',
         'PNC UN EQUITY', 'TRV UN EQUITY', 'KEY UN EQUITY', 'JPM UN EQUITY']

    tickers=['BAC UN EQUITY','BBT UN EQUITY','C UN EQUITY','AMG UN EQUITY','CMA UN EQUITY',
                 'FITB UW EQUITY','HBAN UW EQUITY','JPM UN EQUITY','KEY UN EQUITY','MTB UN EQUITY'
             ,'PBCT UW EQUITY','PNC UN EQUITY','RF UN EQUITY','STI UN EQUITY','USB UN EQUITY','WFC UN EQUITY']

    # tickers = ['AAPL UW EQUITY', 'AMZN UW EQUITY', 'BA UN EQUITY', 'V UN EQUITY',
    #       'FITB UW EQUITY','HBAN UW EQUITY','GS UN EQUITY','JNJ UN EQUITY','JPM UN EQUITY','MSFT UW EQUITY',
    #           'NKE UN EQUITY','PCG UN EQUITY','PM UN EQUITY','TWX UN EQUITY','INTC UW EQUITY',
    #           'WFC UN EQUITY']

    inputs_d = {
        "general_parameters": {
            "base_currency": "USD",
            "bm_security": "S5BANKX INDEX",
            "bm_source": "BLP",
            "capital_base": 100000,
            "data_freq": "Daily",
            "prices_source": "REUTERS",
            "rebalance_freq": "Daily",
            "recommendations_bt": True,
            'securities': {"security_array": tickers},
            #"securities": {"security_index": "S5BANKX INDEX"},
            #"security_list": ['BLK UN EQUITY'],
            'strategy_type': 'TEST',
            "start_date": "2010-11-01T00:00:00", #2017-05-18   #2010-01-01T00:00:00
            "end_date": "2016-06-10T00:00:00"
        },
        "data": {"extra_market_data": ["Open","High","Low","Close","Volume"]}

        #open, high, low, close
    }

    inputs_id  = {
        "general_parameters": {
            "base_currency": "USD",
            "bm_security": "S5BANKX INDEX",
            "bm_source": "BLP",
            "capital_base": 1000000,
            "data_freq": "Minute",
            "prices_source": "IB",
            "rebalance_freq": "Minute",
            "recommendations_bt": True,
            'securities': {"security_array": tickers},
            #'securities': {"security_list": ['WEC UN EQUITY']},
            # 'securities': {'security_index': 'SPX INDEX', 'security_set': 'DJ30'}
            'strategy_type': 'TEST',
            "start_date": "2016-07-08T00:00:00", #2017-05-18   #2010-01-01T00:00:00
            "end_date": "2017-01-04T00:00:00"
        },
        "data": {"extra_market_data": ["Open","High","Low","Close","Volume"]}
    }
    # inputs = heuristrats.parse_db_inputs(inputs_id)


    strat = Strategy_all_stocks(tickers)
    heuristrats.strategy_backtest(
        context_pk=None,
        inputs=inputs_id,
        get_pdf=True,
        substrat=strat
    )


if __name__ == "__main__":
    test_strategy()

