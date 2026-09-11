import glob, os, pandas as pd

for date, label in [('2026-05-27', 'Yesterday (May 27)'), ('2026-05-28', 'Today (May 28)')]:
    print('\n=== ' + label + ' ===')
    files = glob.glob('/home/ubuntu/laplace/logs/trades_*_' + date + '.csv')
    data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) > 1 and 'GrossPnL' in df.columns:
                ticker = os.path.basename(f).split('_')[1]
                trades = len(df)
                gross = df['GrossPnL'].sum()
                brokerage = df['Brokerage'].sum()
                net = df['NetPnL'].sum()
                data.append({'T': ticker, 'N': trades, 'G': gross, 'B': brokerage, 'Net': net})
        except:
            pass
    if not data:
        print('No completed trades found.')
    else:
        total = sum(d['Net'] for d in data)
        data.sort(key=lambda x: x['Net'], reverse=True)
        for d in data:
            line = d['T'].ljust(15) + ' | Trades:' + str(d['N']).rjust(4) + ' | Gross:Rs' + str(round(d['G'],2)).rjust(9) + ' | Brok:Rs' + str(round(d['B'],2)).rjust(7) + ' | Net:Rs' + str(round(d['Net'],2)).rjust(9)
            print(line)
        print('TOTAL Net: Rs' + str(round(total, 2)))
