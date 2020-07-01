import json
from collections import Counter
import numpy as np

data_path='raw_data.json'
with open(data_path,'r') as file:
    raw_data = json.load(file)



acquiring_features=["host_response_rate", "host_acceptance_rate",
               "host_listings_count", "accommodates", "bathrooms", "bedrooms", 
               "beds", "number_of_reviews", "review_scores_rating", 
               "reviews_per_month"]

binarialize_features={'bed_type':{'Real Bed':0,'Futon':1,'Pull-out Sofa':2,'Airbed':3,'Couch':4},
                      'property_type':{'Apartment':0,'House':1,'Condominium':2},
                      'room_type':{'Entire home/apt': 0, 'Private room': 1, 'Shared room': 2},
                      "cancellation_policy":{'strict':0,'flexible': 1, 'moderate': 2, '':3,
                                             'super_strict_30': 4, 'super_strict_60': 5},
                      'host_is_superhost':{'t':1,'f':0}}
binarialize_features['zipcode']={}
testing_features=['zipcode']
fea_dict={}
for fea in testing_features:
    fea_dict[fea]=Counter()
counter=0
summary_counter=0
for row in raw_data['features']:
    counter+=1
    for fea in testing_features:
        fea_dict[fea][row[fea]]+=1
print('num listing:',counter)
counter_zip=0
for key in fea_dict['zipcode']:
    if fea_dict['zipcode'][key]>0:
        binarialize_features['zipcode'][key]=counter_zip
        counter_zip+=1
print('number of zips:',counter_zip)
    
num_features=len(acquiring_features)
for fea in binarialize_features:
    num_features+=len(binarialize_features[fea])

print('num_features:', num_features)
target_feature='price'

included_ids=set()
samples=[]
labels=[]
prev_id_to_new_id = [-1 for i in range(len(raw_data['features']))]
now_id_to_prev_id = []
for (rowid,row) in enumerate(raw_data['features']):
    if row['id'] in included_ids:
        continue
    included_ids.add(row['id'])
    this_sample=np.zeros(num_features)
    not_include=False
    for (i,key) in enumerate(acquiring_features):
        if row[key]=='':
            not_include=True
            break
        if key.endswith('rate'):
            row[key]=row[key].replace('%','')
        this_sample[i]=float(row[key])

    if not_include:
        continue
    current_fea=len(acquiring_features)
    current_fea=len(acquiring_features)
    for key in binarialize_features:
        if row[key] not in binarialize_features[key]:
            not_include=True
            break
        this_sample[current_fea+binarialize_features[key][row[key]]]=1
        current_fea+=len(binarialize_features[key])
    if not_include:
        continue
    prev_id_to_new_id[rowid] = len(samples)
    now_id_to_prev_id.append(rowid)
    samples.append(this_sample.tolist())
    this_price=float(row['price'].replace('$','').replace(',',''))
    labels.append(this_price)



new_label_data = [raw_data['labels'][idx] for idx in now_id_to_prev_id]
new_comp_data =[]
for c_item in raw_data['comparisons']:
    id1,id2=c_item['pair']
    if prev_id_to_new_id[id1]!=-1 and prev_id_to_new_id[id2]!=-1:
        new_comp_data.append({'pair':(prev_id_to_new_id[id1], prev_id_to_new_id[id2]), 'data':c_item['data']})

num_train = prev_id_to_new_id[raw_data['num_train_data']]+1
out_data ={
    'features':samples,
    'original_price':labels,
    'labels':new_label_data,
    'comparisons':new_comp_data,
    'num_train_data':num_train
}
json.dump(out_data, open('vectorized_data.json','w'))
