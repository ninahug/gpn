import pandas as pd
import sys

if __name__ == '__main__':
	df = pd.read_csv('d2n50D1_2opt.csv' , sep = ',', header = None)
	mean_time = sum(df.iloc[:,0].values)/len(df.iloc[:,0])
	mean_time_2opt = sum(df.iloc[:,1].values)/len(df.iloc[:,1])
	mean_cost = sum(df.iloc[:,2].values)/len(df.iloc[:,2])
	
	dfnew = pd.DataFrame([['mean time', 'mean time 2opt', 'mean cost'], [mean_time, mean_time_2opt, mean_cost]])
	df = df.append(dfnew, ignore_index=True)
	print(df)
	df.to_csv('d2n50D1_2opt.csv', header = False, index = False)
	# df.to_csv('new.csv', header = False, index = False)
