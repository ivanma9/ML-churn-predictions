import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from openai import OpenAI
import utils as ut

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ['GROQ_API_KEY'])

def load_model(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)


decision_tree_model = load_model("decision_tree_model.pkl")
random_forest_model = load_model("random_forest_model.pkl")
xgb_model = load_model("xgb_model.pkl")
naive_bayes_model = load_model("gaussian_nb_model.pkl")
svm_model = load_model("svc_model.pkl")
knn_model = load_model("knn_model.pkl")
logistic_regression_model = load_model("logistic_regression_model.pkl")

xgb_model_featuredEng = load_model("xgb_model-featuredEng.pkl")
smote_model = load_model("xgb_model-smote.pkl")
voting_classifier_model = load_model("ensemble_model.pkl")


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_of_products, has_credit_card, is_active_member,
                  estimated_salary):
  input_dict = {
      "CreditScore": credit_score,
      'Age': age,
      'Tenure': tenure,
      'Balance': balance,
      'NumOfProducts': num_of_products,
      'HasCrCard': int(has_credit_card),
      'IsActiveMember': int(is_active_member),
      'EstimatedSalary': estimated_salary,
      'Geography_France': 1 if location == 'France' else 0,
      'Geography_Germany': 1 if location == 'Germany' else 0,
      'Geography_Spain': 1 if location == 'Spain' else 0,
      'Gender_Male': 1 if gender == 'Male' else 0,
      'Gender_Female': 1 if gender == 'Female' else 0
  }

  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def make_prediction(input_df, input_dict):
  probabilities = {
      'Logistic Regressin': logistic_regression_model.predict_proba(input_df)[0][1],
      'XGBoost': xgb_model.predict_proba(input_df)[0][1],
      'NaiveBayes': naive_bayes_model.predict_proba(input_df)[0][1],
      'KNN': knn_model.predict_proba(input_df)[0][1]
  }
  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = ut.create_guage_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The customer has a {avg_probability:.2%} probability of churning")

  with col2:
    fig_probs = ut.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  return avg_probability


def explain_prediction(probability, input_dict, surname):
  prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.
  Your machine learning model has predicted that a customer named {surname} has a {round(probability*100, 1)}% probability of churning based on info provided below. 
  Here is the customer's information:
  {input_dict}
  Here are the machine learning model's top 10 most important features for predicting churn:
  |index|Feature|Importance|
  |---|---|---|
  |4|NumOfProducts|0\.3238884210586548|
  |6|IsActiveMember|0\.16414643824100494|
  |1|Age|0\.10955004394054413|
  |9|Geography\_Germany|0\.09137332439422607|
  |3|Balance|0\.052786167711019516|
  |8|Geography\_France|0\.04646328464150429|
  |11|Gender\_Female|0\.04528257995843887|
  |10|Geography\_Spain|0\.03685470297932625|
  |0|CreditScore|0\.035005148500204086|
  |7|EstimatedSalary|0\.032655227929353714|
  |5|HasCrCard|0\.03194035589694977|
  |2|Tenure|0\.030054276809096336|
  |12|Gender\_Male|0\.0|


  {pd.set_option('display.max_columns', None)}

  Here are summary statistics for churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are summary statistics for non-churned customers:
  {df[df['Exited'] == 0].describe()}

  - If the customer has a over 40% risk of churning generate a 3 sentence explanantion of why they are at risk of churning.
  - If the customer has a less than 40% risk of churning generate a 3 sentence explanantion of why they are not at risk of churning.
  - Your explanation should be based of customer inforamtion, the summary statistics of all customers, and feature importances provided.
  Do not mention probability of churning or the machine learning model. Do not mention passively of what you are going to do, Just explain the prediction in paragraph form.
  """

  print("EXPLANAION PROMPT", prompt)

  raw_response = client.chat.completions.create(
    model="llama-3.2-3b-preview",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ]
  )
  return raw_response.choices[0].message.content

st.title("Customer Churn Prediction")
df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId'] } - {row['Surname']}" for _, row in df.iterrows()
]
selected_customer = st.selectbox("Select a customer", customers)

if selected_customer:
  selected_customer_id = int(selected_customer.split(" - ")[0])
  print("Select customer id: ", selected_customer_id)

  selected_surname = selected_customer.split(" - ")[1]
  print("Select customer surname: ", selected_surname)
  selected_customer_row = df.loc[df["CustomerId"] ==
                                 selected_customer_id].iloc[0]
  print(selected_customer_row)

  col1, col2 = st.columns(2)

  with col1:
    credit_score = st.number_input("Credit Score",
                                   min_value=300,
                                   max_value=850,
                                   value=int(
                                       selected_customer_row["CreditScore"]))
    location = st.selectbox("Location", ["Spain", "France", "Germany"],
                            index=["Spain", "France", "Germany"
                                   ].index(selected_customer_row["Geography"]))

    gender = st.radio(
        "Gender", ["Male", "Female"],
        index=0 if selected_customer_row["Gender"] == "Male" else 1)

    age = st.number_input("Age",
                          min_value=18,
                          max_value=100,
                          value=int(selected_customer_row["Age"]))

    tenure = st.number_input("Tenure (yrs)",
                             min_value=0,
                             max_value=50,
                             value=int(selected_customer_row["Tenure"]))

  with col2:
    balance = st.number_input("Balance",
                              min_value=0.0,
                              value=float(selected_customer_row["Balance"]))

    num_products = st.number_input("Number of Products",
                                   min_value=1,
                                   max_value=10,
                                   value=int(
                                       selected_customer_row["NumOfProducts"]))

    has_credit_card = st.checkbox("Has Credit Card",
                                  value=bool(
                                      selected_customer_row["HasCrCard"]))

    is_active_member = st.checkbox(
        "Is Active Member",
        value=bool(selected_customer_row["IsActiveMember"]))

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer_row['EstimatedSalary']))

  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, num_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)

  avg_prob = make_prediction(input_df, input_dict)

  explanation = explain_prediction(avg_prob, input_dict, selected_customer_row["Surname"])

  st.markdown("----")
  st.subheader("Explanation of Prediction")
  st.markdown(explanation)
