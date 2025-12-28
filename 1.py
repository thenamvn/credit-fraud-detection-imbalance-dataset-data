#load csv and read is_fraud column have many values is 0
import pandas as pd
df = pd.read_csv('fraudTrain.csv')
fraud_count = df['is_fraud'].value_counts().get(1, 0)
print(fraud_count)
