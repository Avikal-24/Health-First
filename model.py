import numpy as np
import pandas as pd
import pickle

#list of symptoms:
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']
#print(len(l1))

#list of diseases/outputs
disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)
# TRAINING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

#print(df.tail())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y) #np.ravel() flattens a 2d matrix into a 1d line of data
#print(y)

# TESTING DATA ts --------------------------------------------------------------------------------
ts=pd.read_csv("testing.csv")
ts.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= ts[l1]
y_test = ts[["prognosis"]]
np.ravel(y_test)
#print(y_test.head())

# ------------------------------------------------------------------------------------------------------

#def DecisionTree():

from sklearn import tree

model_dt = tree.DecisionTreeClassifier()  # empty model of the decision tree
model_dt = model_dt.fit(X, y)

# calculating accuracy-------------------------------------------------------------------
from sklearn.metrics import accuracy_score

y_pred = model_dt.predict(X_test)
print(accuracy_score(y_test, y_pred)) #normalised accuracy
print(accuracy_score(y_test, y_pred, normalize=False))
# -----------------------------------------------------

Symptom1='chest_pain'
Symptom2='drying_and_tingling_lips'
Symptom3='dizziness'
Symptom4='family_history'
Symptom5='fluid_overload'

# Symptom1='chest_pain'
# Symptom2='chest_pain'
# Symptom3='chest_pain'
# Symptom4='chest_pain'
# Symptom5='chest_pain'

psymptoms = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]

for k in range(0, len(l1)):
    for z in psymptoms:
        if (z == l1[k]):
            l2[k] = 1

inputtest = [l2]
predict = model_dt.predict(inputtest)
predicted = predict[0]

h = 'no'
for a in range(0, len(disease)):
    if (predicted == a):
        h = 'yes'
        break
if (h == 'yes'):
    output=disease[a]
else:
    output='NOT FOUND'

print(output)

#save model as a pickle file (it serialises the python object as a stream of bits and store on disk
pickle.dump(model_dt, open('model.pkl','wb')) #wb is write binary






# HOGYA! ,predicting outputs from a pickeled file
# Loading model from pickled file to compare the results
model = pickle.load(open('model.pkl','rb')) #rb=read binary
psymptoms = ['lack_of_concentration','increased_appetite','history_of_alcohol_consumption','loss_of_smell','excessive_hunger']

for k in range(0, len(l1)):
    for z in psymptoms:
        if (z == l1[k]):
            l2[k] = 1

inputtest = [l2]
predict = model.predict(inputtest)
predicted = predict[0]

h = 'no'
for a in range(0, len(disease)):
    if (predicted == a):
        h = 'yes'
        break
if (h == 'yes'):
    output=disease[a]
else:
    output='NOT FOUND'

print(output)
#print(model.predict(inputtest))