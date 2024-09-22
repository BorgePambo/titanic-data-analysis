import pandas as pd
import pandas as pd
from pathlib import Path
import os
import pyarrow.parquet as pa

try:
    csv_file = Path('../data/raw/titanic.csv')
    df = pd.read_csv(csv_file)

    #print(df.info())
   
    df['Age'] = df['Age'].fillna(0).astype('int32')

    ##Updating values of table Embarket
    df['Embarked'] = df['Embarked'].replace({
        'C': 'Cherbourg',
        'Q': 'Queenstown',
        'S': 'Southampton'
    })

    df['Embarked'] = df['Embarked'].fillna('Unknown')

    #print(df)
    values = df['Embarked'].unique()
    #print(values)


    #Atualizando os valores da coluna 'Embarket'
    def adicionando_coluna(line):
        if line['Embarked'] == 'Cherbourg':
            return 'France'
        elif line['Embarked'] == 'Queenstown':
            return 'Ireland'
        elif line['Embarked'] == 'Southampton':
            return 'UK'
        else:
            return 'Unknown'  # Opcional, caso haja outro valor inesperado

    # Aplicando a função e criando a nova coluna 'Country'
    df['Country'] = df.apply(adicionando_coluna, axis=1)

    
    values = df['Country'].unique()

    df['Title'] = df['Name'].str.extract(r'(\bMr\b|\bMrs\b|\bMiss\b)', expand=False)
   
    df['Title'] = df['Title'].fillna(df['Sex'].map({'male': 'Mr', 'female': 'Miss'}))


    df['Cleaned_Name'] = df['Name'].str.replace(r' (Mr\.|Mrs\.|Miss\.|Master\.) ', ' ', regex=True)
    
    df.drop(columns=["Name"], inplace=True)

    df.rename(columns={'Cleaned_Name': 'Full Name'}, inplace=True)

    dframe = df[
        ['PassengerId', 
        'Full Name', 
        'Title', 
        'Sex', 
        'Age', 
        'Pclass', 
        'Ticket', 
        'Cabin', 
        'SibSp', 
        'Parch', 
        'Fare', 
        'Survived', 
        'Embarked', 
        'Country']
    ]

    dframe['Cabin'] = dframe['Cabin'].fillna('Unknown')

    # x = dframe['Cabin'].unique()
    # print(dframe['Cabin'].nunique())
    
    # Criar um filtro para as cabines com mais de 2 ocorrências
    cabin_counts = dframe['Cabin'].value_counts()
    popular_cabins = cabin_counts[cabin_counts > 2]
    

    out_dir = os.path.join('../data/clean')
    os.makedirs(out_dir, exist_ok=True)

    #salvando o dataframe em um arquivo json
    json_file = os.path.join(out_dir, 'data.json')
    dframe.to_json(json_file, orient='records', lines=True)
    
    #salvando o dataframe em um arquivo parquet
    parquet_file = os.path.join(out_dir, 'data.parquet')
    dframe.to_parquet(parquet_file, engine='pyarrow')

    print('-----------------------------')


except Exception as ex:
    print(f"Error loading: {ex}")

