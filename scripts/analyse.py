import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


try:
    # Carregando o DataFrame do arquivo Parquet
    df = pd.read_parquet('../data/clean/data.parquet')
    
    # Exibindo o DataFrame no Streamlit
    st.header('Titanic Analyses')
    st.dataframe(df)

    st.write('Parch (Parents/Children):')
    aggreg_parch = df.groupby('Sex').agg(
        minimun_Parch=('Parch', 'min'),
        maximun_Parch=('Parch', 'max'),
        avarage_Parch=('Parch', 'mean'),
        total_Parch=('Parch', 'sum')
    )
    st.dataframe(aggreg_parch)

    st.write("SibSp (siblings and spouses)!")
    aggreg_sibsp = df.groupby('Sex').agg(
        minimun_SibSp=('SibSp', 'min'),
        maximun_SibSp=('SibSp', 'max'),
        avarage_SibSp=('SibSp', 'mean'),
        total_SibSp=('SibSp', 'sum')
    )
    st.dataframe(aggreg_sibsp)

    st.subheader('Analysis of measure')

    col1, col2, col3 = st.columns(3)
    with col1:
        total_persons = len(df['PassengerId'])
        st.metric("Number of Passengers", total_persons)
    with col2:
        dead_person = df[df['Survived'] == 0].shape[0]  
        st.metric("Total of deceased", dead_person)
    with col3:
        survived_person = df[df['Survived'] == 1].shape[0]  
        st.metric("Total of survivor", survived_person)

    
    col4, col5, col6 = st.columns(3)

    with col4:
        male_count = df[df['Sex'] == 'male'].shape[0]
        st.metric(label='Count of Male', value=male_count)

    with col5:
        famele_count = df[df['Sex'] == 'female'].shape[0]
        st.metric(label='Count of Famele', value=famele_count)
    
    with col6:
        mrs_count = df[df['Title'] == 'Mrs'].shape[0]
        st.metric(label="Count of Married", value=mrs_count)

    
    col7, col8, col9 = st.columns(3)

    with col7:
        class_a = df[df['Pclass'] == 1].shape[0]
        st.metric(label='Passenger in class A', value=class_a)

    with col8:
        class_b = df[df['Pclass'] == 2].shape[0]
        st.metric(label='Passenger in class B', value=class_b)
    
    with col9:
        class_c = df[df['Pclass'] == 3].shape[0]
        st.metric(label='Passenger in class C', value=class_c)


    fig = px.bar(df, x='Country', y='PassengerId', title='Passengers onboard by region')
    st.plotly_chart(fig)

    class_map = {1: 'First Class', 2: 'Second Class', 3: 'Third Class'}
    df['Pclass_Label'] = df['Pclass'].map(class_map)

    fig_class = px.bar(df, x='Pclass', y='PassengerId', 
                   title='Passengers onboard by Class 1, 2, 3')
    st.plotly_chart(fig_class)


    
    agg_data = df.groupby('Pclass').agg(Total_Fare=('Fare', 'sum'), Average_Fare=('Fare', 'mean')).reset_index()
    # Criando a tabela com Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Class', 'Total Fare', 'Average Fare']),
        cells=dict(values=[agg_data['Pclass'], agg_data['Total_Fare'], agg_data['Average_Fare']])),
    ])
    # Exibindo a tabela no Streamlit
    st.plotly_chart(fig)

    st.subheader('Number of Passenger by Cabin')
    cabin_counts = df['Cabin'].value_counts().reset_index()
    cabin_counts.columns = ['Cabin', 'Counting Passenger']
    st.dataframe(cabin_counts)

    col11, col12 = st.columns(2)

    with col11:
        adult = df[df['Age'] >= 18].value_counts().shape[0]
        st.metric(label='Maior de Idade', value=adult)

    with col12:
        minor = df[df['Age'] < 18].value_counts().shape[0]
        st.metric(label='Menor de Idade', value=minor)



except Exception as ex:
    st.error(f"Ocorreu um erro: {ex}")
    