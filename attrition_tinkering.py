import numpy as np
import pandas as pd
import sqlite3
import pickle


with open('model.pkl', 'rb') as file:
    output = pickle.load(file)

with sqlite3.connect('nudge_test.db') as conn:
    df = pd.read_sql_query('Select * From user_tp_attrition', conn)

model = output["model"]
binarizer = output["binarizer"]
model.predict(pd.concat((pd.DataFrame(binarizer.fit_transform(df['department'])),
              df["nb_of_sessions"]), axis=1))
output["coefficient_names"]
model.predict_proba(np.array([1, 0, 0, 0, 1]).reshape(1, -1))
