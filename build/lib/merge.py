"""Downstream analysis after bee detection and classification"""


import glob
import pandas as pd
import os
import sys

def merge(CSVdir, dataDir):
    if not os.path.exists(CSVdir):
    	print(CSVdir + ' does not exist')
    if not os.path.exists(dataDir):
    	print(dataDir + ' does not exist')
    out = pd.DataFrame()
    print(glob.glob(os.path.join(CSVdir, '*bees.csv')))
    for csv in glob.glob(os.path.join(CSVdir, '*bees.csv')):
        print(csv)
        df = pd.read_csv(csv)
        path = df['originalFile'][0]
        parent = path.split('stills')[0].split(os.path.basename(dataDir))[1]
        df['parent'] = parent
        df['unit'] = '_'.join(os.path.basename(csv).split('_')[0:-6])
        a, b, c, d, date = os.path.basename(csv).split('_')[-6:-1]
        
        df['camera'] = '_'.join([a, b, c, d])
        df['unitCamera'] = '_'.join(os.path.basename(csv).split('_')[0:-6]+[a, b, c, d])
        df['date'] = date

        out = pd.concat([out, df], axis=0, ignore_index=True)

    out.to_csv('merged.csv')
    return out

def wrangle(merged):
    split = merged['filename'].str.split('_', expand=True)
    merged['time'] = split[3] + split[4]
    timeSplit = split[3].str.split('', expand=True).astype(str)
    merged['hm'] = timeSplit[1] + timeSplit[2] + ':' + timeSplit[3] + timeSplit[4]

    return merged

def temp(wrangled, dataDir):
    tempCSVs = glob.glob(os.path.join(dataDir, '*/*/tempProbes/*.csv'))
    out = pd.DataFrame()
    for t in tempCSVs:
        print(t)
        temps = pd.read_csv(t)
        temps['date'] = pd.to_datetime(temps.date)
        temps['date'] = temps['date'].dt.strftime('%y%m%d')
        temps['hm'] = temps['time']
        temps['tempFile'] = t
        parent = t.split('tempProbes')[0].split(os.path.basename(dataDir))[1]
        corr = wrangled[wrangled['parent'] == parent]
        addtemp = pd.merge(temps, corr, on=['hm', 'date'])
        out = pd.concat([out, addtemp], ignore_index=True)
    
    
    out.to_csv('withTemp.csv')

    return 0

def main(args):
    CSVdir = args[1]
    dataDir = args[2]

    if CSVdir[-1] == '/':
    	CSVdir = CSVdir[:-1]    
    if dataDir[-1] == '/':
    	dataDir = dataDir[:-1]
    merged = merge(CSVdir, dataDir)
    wrangled = wrangle(merged)
    temp(wrangled, dataDir)

if __name__ == "__main__":
	status = main(sys.argv)
	sys.exit(status)