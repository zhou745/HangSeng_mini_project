# install the package
pip install -e .

# data prepareration
run: python data_process/down_load_data.py

# main result

train:

python main_candlestick.py -n main_result --base configs/HK_5M_r_v7.yaml -t --root_dir logs --gpus [your gpu here]

test:
python test_tools/compute_ic_on_val.py
