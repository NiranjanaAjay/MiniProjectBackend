# import pandas as pd
# import numpy as np
# import random

# def generate_patient():
#     disease = random.choice([
#         "Tuberculosis",
#         "Leukemia",
#         "Bone_Marrow_Failure",
#         "Chronic_Kidney_Disease",
#         "Liver_Failure",
#         "Severe_Anemia",
#         "No_Critical_Disease"
#     ])

#     age = random.randint(18, 75)
#     gender = random.choice([0, 1])  # 0 = Female, 1 = Male

#     # Default normal values
#     fever = 0
#     cough = 0
#     breathing_issue = 0
#     fatigue = 0
#     weight_loss = 0

#     systolic_bp = random.randint(110, 130)
#     diastolic_bp = random.randint(70, 85)
#     cholesterol = random.randint(150, 220)
#     hemoglobin = round(random.uniform(12, 16), 1)
#     wbc_count = random.randint(4000, 11000)
#     platelet_count = random.randint(150000, 450000)
#     creatinine = round(random.uniform(0.6, 1.2), 2)
#     bilirubin = round(random.uniform(0.3, 1.2), 2)

#     if disease == "Tuberculosis":
#         fever = cough = breathing_issue = weight_loss = 1
#         wbc_count = random.randint(12000, 18000)

#     elif disease == "Leukemia":
#         fever = fatigue = 1
#         wbc_count = random.randint(25000, 60000)
#         hemoglobin = round(random.uniform(5, 8), 1)
#         platelet_count = random.randint(20000, 90000)

#     elif disease == "Bone_Marrow_Failure":
#         hemoglobin = round(random.uniform(5, 8), 1)
#         wbc_count = random.randint(2000, 3500)
#         platelet_count = random.randint(20000, 90000)

#     elif disease == "Chronic_Kidney_Disease":
#         fatigue = 1
#         creatinine = round(random.uniform(2.5, 6), 2)
#         systolic_bp = random.randint(150, 180)

#     elif disease == "Liver_Failure":
#         fatigue = weight_loss = 1
#         bilirubin = round(random.uniform(2.5, 6), 2)

#     elif disease == "Severe_Anemia":
#         fatigue = 1
#         hemoglobin = round(random.uniform(5, 8), 1)

#     return [
#         age, gender, fever, cough, breathing_issue, fatigue, weight_loss,
#         systolic_bp, diastolic_bp, cholesterol,
#         hemoglobin, wbc_count, platelet_count,
#         creatinine, bilirubin, disease
#     ]

# columns = [
#     "age", "gender", "fever", "cough", "breathing_issue", "fatigue", "weight_loss",
#     "systolic_bp", "diastolic_bp", "cholesterol",
#     "hemoglobin", "wbc_count", "platelet_count",
#     "creatinine", "bilirubin", "target_disease"
# ]

# data = [generate_patient() for _ in range(1000)]

# df = pd.DataFrame(data, columns=columns)
# df.to_csv("careconnect_synthetic_dataset.csv", index=False)

# print("Dataset generated successfully!")

import pandas as pd
import random

def generate_patient():
    disease = random.choice([
        "Leukemia",
        "Aplastic_Anemia",
        "Thalassemia_Major",
        "Chronic_Kidney_Disease",
        "Liver_Cirrhosis",
        "Liver_Failure",
        "No_Critical_Disease"
    ])

    age = random.randint(18, 75)
    gender = random.choice([0, 1])

    fever = 0
    fatigue = 0
    weight_loss = 0
    systolic_bp = random.randint(110, 130)
    diastolic_bp = random.randint(70, 85)
    hemoglobin = round(random.uniform(12, 16), 1)
    wbc_count = random.randint(4000, 11000)
    platelet_count = random.randint(150000, 450000)
    creatinine = round(random.uniform(0.6, 1.2), 2)
    bilirubin = round(random.uniform(0.3, 1.2), 2)

    if disease == "Leukemia":
        fever = 1
        fatigue = 1
        wbc_count = random.randint(30000, 60000)
        hemoglobin = round(random.uniform(5, 8), 1)
        platelet_count = random.randint(20000, 90000)

    elif disease == "Aplastic_Anemia":
        fatigue = 1
        wbc_count = random.randint(2000, 3500)
        hemoglobin = round(random.uniform(5, 8), 1)
        platelet_count = random.randint(20000, 90000)

    elif disease == "Thalassemia_Major":
        fatigue = 1
        hemoglobin = round(random.uniform(5, 7), 1)
        wbc_count = random.randint(5000, 12000)
        platelet_count = random.randint(150000, 300000)

    elif disease == "Chronic_Kidney_Disease":
        fatigue = 1
        creatinine = round(random.uniform(2.5, 6), 2)
        systolic_bp = random.randint(150, 180)

    elif disease == "Liver_Cirrhosis":
        fatigue = 1
        weight_loss = 1
        bilirubin = round(random.uniform(1.5, 3), 2)
        platelet_count = random.randint(80000, 140000)

    elif disease == "Liver_Failure":
        fatigue = 1
        weight_loss = 1
        bilirubin = round(random.uniform(3.5, 7), 2)
        creatinine = round(random.uniform(1.5, 3), 2)

    return [
        age, gender, fever, fatigue, weight_loss,
        systolic_bp, diastolic_bp,
        hemoglobin, wbc_count, platelet_count,
        creatinine, bilirubin, disease
    ]

columns = [
    "age", "gender", "fever", "fatigue", "weight_loss",
    "systolic_bp", "diastolic_bp",
    "hemoglobin", "wbc_count", "platelet_count",
    "creatinine", "bilirubin", "target_disease"
]

data = [generate_patient() for _ in range(1200)]
df = pd.DataFrame(data, columns=columns)
df.to_csv("careconnect_synthetic_dataset_v2.csv", index=False)

print("New dataset generated successfully!")