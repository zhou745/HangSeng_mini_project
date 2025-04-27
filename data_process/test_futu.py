import futu as ft

def main():
    quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data, _ = quote_ctx.request_history_kline('HK.00020',ktype='K_1M', start='2024-09-11', end='2024-09-11',
                                                              max_count=1000)

    quote_ctx.close()
    print(data)
if __name__ == '__main__':
    main()