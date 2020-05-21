# ITS with LSTM method.


## Code structure
```
.
├── README.md
├── code
│   ├── __pycache__
│   ├── calculate_distance.py
│   ├── calculate_single_station_score.py
│   ├── create_graph_matrix.py
│   ├── create_graph_matrix_distance.py
│   ├── create_tgcn_input.py
│   ├── find_station_info.py
│   ├── load_data.py
│   ├── load_net.py
│   ├── lstm_01.py
│   ├── lstm_02.py
│   ├── lstm_03.py
│   ├── lstm_04.py
│   ├── lstm_05.py
│   ├── lstm_06.py
│   ├── lstm_test_01.py
│   ├── lstm_test_02.py
│   ├── lstm_test_03_single_station_mac.py
│   ├── lstm_train_01.py
│   ├── lstm_train_02.py
│   ├── lstm_train_02_single_station.py
│   ├── lstm_train_03.py
│   ├── lstm_train_03_single_station.py
│   ├── lstm_train_04.py
│   ├── lstm_train_04_single_station.py
│   ├── lstm_train_05_single_station_mac.py
│   ├── lstm_train_06_single_station_mac.py
│   ├── net_lstm_2.pkl
│   ├── net_lstm_all_station.pkl
│   ├── plot
│   ├── step0_metroFlow.py
│   ├── test_server.py
│   └── utils.py
├── data
│   ├── raw_data
│   └── true_data
├── model
│   ├── net_demo.pkl
│   ├── net_lstm.pkl
│   ├── net_lstm_2.pkl
│   └── single_station_model
├── plot_cn_paper
│   ├── plot_all_station_error.py
│   ├── plot_single_station_error.py
│   ├── plot_single_station_training_mape_mac.py
│   └── reload_model_to_plot_flow.py
└── result
    ├── all_station_result
    ├── early_early_result
    ├── early_result
    ├── single_station_result
    ├── transferStation_info.csv
    └── transferStation_info_?\225??\220\206?\220\216.xlsx

```
