'''
Create table with schema sid, # app, # true app, attribute, attribute value, 
can join on table with school district and school info, and table with zip code to 
residential district info and boroughs. Then, we can do dynamic queries to generate rankings 
based on different conditions
So the differences in generated rankings by these queries can help inform how we model 
preferences (Mallows)
'''
import pandas as pd
data_path =  "master_data.xlsx"
df = pd.read_excel(data_path)
df2 = df[df['School DBN'] == '02M296']
print(df2['Residential District'].value_counts())
df2 = df[df['Residential District'] == 3]
print(len(df2))
print(df2.sample(5))
'''
Okay so we can see the Residential District view has been set. What we
have not done is language and zip code.

Plan: Generate many ranking distributions based on views and present them

Per view, get ranking of true : total applicant ratio

1. Residential District
2. Language
3. Zip Code
4. Borough

We can see distributions of rankings per view

And then we can build a Mixture of Mallows models with these as centers?
That satisfy match statistics in DATA3
We are ignoring DATA2
We are using DATA1 mainly, with school info from DATA4.

'''