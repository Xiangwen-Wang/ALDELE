import pandas as pd

# read five model results
df1 = pd.read_csv('Model1.csv')
df2 = pd.read_csv('Model2.csv')
df3 = pd.read_csv('Model3.csv')
df4 = pd.read_csv('Model4.csv')
df5 = pd.read_csv('Model5.csv')


dataset1_a = df1.iloc[-1][['RMSE_dev', 'RMSE_test']].mean()
dataset2_a = df2.iloc[-1][['RMSE_dev', 'RMSE_test']].mean()
dataset3_a = df3.iloc[-1][['RMSE_dev', 'RMSE_test']].mean()
dataset4_a = df4.iloc[-1][['RMSE_dev', 'RMSE_test']].mean()
dataset5_a = df5.iloc[-1][['RMSE_dev', 'RMSE_test']].mean()

# compared the average RMSE of each model and find the optimal.
if dataset1_a.mean() < dataset2_a.mean() and dataset1_a.mean() < dataset3_a.mean() and dataset1_a.mean() < dataset4_a.mean() and dataset1_a.mean() < dataset5_a.mean():
    print("The optimal model is Model1")
elif dataset2_a.mean() < dataset1_a.mean() and dataset2_a.mean() < dataset3_a.mean() and dataset2_a.mean() < dataset4_a.mean() and dataset2_a.mean() < dataset5_a.mean():
    print("The optimal model is Model2")
elif dataset3_a.mean() < dataset1_a.mean() and dataset3_a.mean() < dataset2_a.mean() and dataset3_a.mean() < dataset4_a.mean() and dataset3_a.mean() < dataset5_a.mean():
    print("The optimal model is Model3")
elif dataset4_a.mean() < dataset1_a.mean() and dataset4_a.mean() < dataset2_a.mean() and dataset4_a.mean() < dataset3_a.mean() and dataset4_a.mean() < dataset5_a.mean():
    print("The optimal model is Model4")
else:
    print("The optimal model is Model5")