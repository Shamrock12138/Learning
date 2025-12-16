import pandas as pd

df = pd.read_csv(r'f:\GraduateStudent\Data\Airline_Passengers\cleaned_airline_passengers.csv', encoding='utf-8')
print(df)
df['date'] = pd.to_datetime(df['date'])
print(df)
df = df.sort_values('date')
value_col = [col for col in df.columns if col != 'date'][0]
print(value_col)
raw_data = df[value_col].values
print(raw_data)