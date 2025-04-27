import os
import futu as ft
import pandas as pd
from tqdm import tqdm
from datetime import date, timedelta
import time

#Indices - Index Constituents - Hang Seng Index
code_list = ['HK.00001','HK.00002','HK.00003','HK.00005','HK.00006',
             'HK.00011','HK.00012','HK.00016','HK.00027','HK.00066',
             'HK.00101','HK.00175','HK.00241', 'HK.00267', 'HK.00285',
             'HK.00291', 'HK.00316', 'HK.00322', 'HK.00386', 'HK.00388',
             'HK.00669','HK.00688','HK.00700','HK.00762','HK.00823',
             'HK.00836','HK.00857','HK.00868','HK.00881','HK.00939',
             'HK.00941','HK.00960','HK.00241', 'HK.00968', 'HK.00981',
             'HK.01024', 'HK.01038', 'HK.01044', 'HK.01088', 'HK.01093',
             'HK.01099','HK.01109','HK.01113','HK.01177','HK.01209',
             'HK.01211','HK.01378','HK.01398','HK.01810','HK.01876',
             'HK.01928','HK.01929','HK.01997', 'HK.02020', 'HK.02269',
             'HK.02313', 'HK.02319', 'HK.02331', 'HK.02359', 'HK.02382',
             'HK.02628','HK.02688','HK.02899','HK.03690','HK.03968',
             'HK.03988','HK.06618','HK.06862','HK.09618','HK.09888',
             'HK.09901','HK.09988','HK.09999'
             ]
#
# code_list = [ '6Emain','6Bmain','6Jmain','6Smain','6Amain',
#               '6Cmain','6Nmain','6Mmain','M6Emain','M6Amain',
#               'M6Bmain','MIRmain','MSFmain','MCDmain','6Lmain'
#              ]

def main():
    save_root = "HK_5M"
    # save_root = "Option_5M"
    record = 400 #331  #66 for 5M 331 for 1M
    K_type = "K_5M" #"K_5M"

    for code in tqdm(code_list):
        quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
        os.makedirs(f"./data/{save_root}/{code}", exist_ok=True)
        count = 73

        # start_date = date(2025, 1, 1)
        # end_date = date(2025, 4, 22)
        start_date = date(2025, 4, 23)
        end_date = date(2025, 4, 24)
        delta = timedelta(days=1)
        while start_date <= end_date:
            start = start_date.strftime("%Y-%m-%d")
            end = start



            ret, data, page_req_key = quote_ctx.request_history_kline(code,ktype=K_type, start=start, end=end,
                                                                  max_count=record)

            if isinstance(data,str):
                print(data)
                break
            if data.size>0:
                data.to_pickle(f"./data/{save_root}/{code}/{count}.pkl")
                count+=1

            start_date += delta
            #intenstionally sleep 0.5 a second
            time.sleep(0.5)

        quote_ctx.close()
        print(f"finished code {code}")

if __name__ == '__main__':
    main()