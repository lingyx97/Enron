#!/usr/bin/python

import sys
import pickle
import operator
sys.path.append("../tools/")
import numpy as np
from pprint import pprint
from tester import dump_classifier_and_data
from collections import defaultdict
from collections import Counter
from support_f import find_outliers, formatting_data, \
	find_best_PCA, find_best_Kmeans

PRINT_OUTLIERS=False
CHECK_NAN=False
FIND_N_FOR_PCA=False
FIND_N_FOR_KMEANS=False
FIND_BEST_CLASSIFIER=False
SCORE_FEATURES=False

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
# Task 1: Select what features you'll use.
feature_list=[]
for i in data_dict.keys():
	feature_list+=data_dict[i].keys()

feature_list=list(set(feature_list)-set(["poi","email_address"]))
feature_list.insert(0,"poi")
# pprint(data_dict)

# Task 2: Remove outliers
nan_feat=defaultdict(int)
nan_peop=defaultdict(int)
for j in list(set(feature_list)-set(["poi"])):
	#intepret "NaN" as np.nan
	for i in data_dict.keys():
		if data_dict[i][j]!="NaN":
			data_dict[i][j]=float(data_dict[i][j])
		else:
			data_dict[i][j]=np.nan
			if CHECK_NAN:
				#record the number of nan's
				nan_feat[j]+=1
				nan_peop[i]+=1
	#print out outliers
	if PRINT_OUTLIERS:
		print(find_outliers(data_dict,j))

#the key 'TOTAL' will be removed, and all other values will be kept
del data_dict["TOTAL"]
nan_feat=dict(nan_feat)

if CHECK_NAN:
	for i in nan_feat:
		#calculate proportion of nan's
		nan_feat[i]=nan_feat[i]/len(data_dict.keys())
	nan_feat = sorted(nan_feat.items(), key=operator.itemgetter(1))
	for i in nan_peop:
		#print entries which have all fields except poi to be nan's
		if nan_peop[i]>=len(feature_list)-1:
			print(i)
	pprint(nan_feat)
	print(nan_peop)

#six features have over half missing values,
#loan_advances contains almost no valid value, so it will be removed
#other five features will have their nan value replaced by 0
#LOCKHART EUGENE E has all features as NAN, and he is not a poi, so he will be removed
#THE TRAVEL AGENCY IN THE PARK is also a strange entry by inspection, which will be removed
del data_dict["LOCKHART EUGENE E"]
del data_dict["THE TRAVEL AGENCY IN THE PARK"]
feature_list.remove("loan_advances")
for i in ["director_fees","restricted_stock_deferred","deferral_payments",\
	"deferred_income","long_term_incentive"]:
	for j in data_dict:
		if data_dict[j][i]==np.nan:
			data_dict[j][i]=0


# # Task 3: Create new feature(s)
for i in data_dict:
	#calculate the percentage of poi related emails among all emails sent and received
	data_dict[i]["pct_from_poi"]=\
		data_dict[i]["from_poi_to_this_person"]/data_dict[i]["to_messages"]
	data_dict[i]["pct_to_poi"]=\
		data_dict[i]["from_this_person_to_poi"]/data_dict[i]["from_messages"]

#remove unnecessary features and add in new ones
feature_list=list(set(feature_list)-set(["from_this_person_to_poi",
	"to_messages","from_poi_to_this_person","from_messages"]))
feature_list+=["pct_from_poi","pct_to_poi"]

features,labels=formatting_data(data_dict,feature_list)

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest,  f_classif
from sklearn.cross_validation import ShuffleSplit
'''
from sklearn.svm import SVC
clf=SVC()
'''
#intended to decide best PCA by SVC, but found the score too high.
#printing out one of the predictions we can see that SVC is predicting 
#	all labels as 0.
#use Decision Trees instead
clf=DecisionTreeClassifier(min_samples_split=4)

if SCORE_FEATURES==False:
	for k in range(100):
		scores=[]
		skb=SelectKBest(f_classif)
		skb.fit(features,labels)
		scores.append(skb.scores_)
	print(sorted(list(set(feature_list)-set(["poi"]))))
	print(np.mean(scores,axis=0,dtype=np.float64))

if FIND_N_FOR_PCA:
	tmp_list=[]
	#iterate 200 times and store the best n_components each time
	for i in range(200):
		tmp_list+=find_best_PCA(features,labels,clf)
	print(Counter(tmp_list))
	#index zero appears most oftenly, and 4 follows 0.
	#index zero=1, which although not nonsense but still sounds unintuitive
	#so I will use both 1 and 5 for next parts.

for i in [1,5]:
	#use the best PCA found above
	pca=PCA(n_components=i)
	features_pca=pca.fit_transform(features)
	if FIND_N_FOR_KMEANS:
		tmp_list=[]
		#iterate 200 times and store the best n_clusters each time
		for i in range(200):
			print(str(i/200)+"\r") #a progress indicator
			tmp_list+=find_best_Kmeans(features_pca,labels,clf)
		print(Counter(tmp_list))
	#will choose 12 for n_components=1
	# 3 for n_component=5



# # Task 4: Try a varity of classifiers
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, fbeta_score


if FIND_BEST_CLASSIFIER:
	scores=[]
	for trans_param in [[1,12],[5,3]]:
		clf=[]
		clf_score=defaultdict(list)
		clf_F1=defaultdict(list)
		clf_F2=defaultdict(list)
		#use the best PCA and Kmeans found above
		pca=PCA(n_components=trans_param[0])
		kmc=KMeans(n_clusters=trans_param[1])
		features_tmp=pca.fit_transform(features)
		features_tmp=kmc.fit_transform(features_tmp)
		#define three possibly good classifiers
		gnb = GaussianNB()
		svc= SVC()
		svc_param={
			"kernel":["rbf","poly"],
			"C":[0.25,0.5,1,2,4,8],
			"degree":[i+1 for i in range(10)],
			"gamma":[0.05,0.1,0.2,0.4,1]
			}
		abc=AdaBoostClassifier()
		abc_param={"n_estimators" : [10,25,50,75,100],
			"learning_rate" :   [0.5,1,2]}
		#store all three classifiers into a list to make iteration easier
		#use grid search to find the best arguments. Tuning with F1 score
		clf=[gnb]
		clf.append(GridSearchCV(svc,svc_param,scoring='f1'))
		clf.append(GridSearchCV(abc,abc_param,scoring='f1'))
		#generate training and testing sets
		ss= ShuffleSplit(len(features_tmp),10, test_size=0.5)
		nn=0
		for a,b in ss:
			feat_train,feat_test,lab_train,lab_test=features_tmp[a],features_tmp[b],\
				labels[a],labels[b]
			#score with each classifier
			for i in range(3):
				clf[i].fit(feat_train,lab_train)
				clf_score[i].append(accuracy_score(lab_test,clf[i].predict(feat_test)))
				clf_F1[i].append(f1_score(lab_test,clf[i].predict(feat_test)))
				clf_F2[i].append(fbeta_score(lab_test,clf[i].predict(feat_test),2))
		#find the median score of each classifier
		for i in clf_score:
			scores.append((np.median(clf_score[i]),np.median(clf_F1[i]),np.median(clf_F2[i])))
	print(scores)
	print("\n")
	#Gaussian Naive Bayes performs the best in both case, 
	#and all classifiers have higher F1 score in first case. i.e.[1,12]
	#Although some classifiers have higher accuracies in the second case
	#The following will be the final combination

# Task 5: Tune your classifier to achieve better than .3 precision and recall
from sklearn.preprocessing import Imputer, MinMaxScaler
#combine all together
imp=Imputer(missing_values=np.nan,strategy="median")	
mms=MinMaxScaler()
pca=PCA(n_components=1)
kmc=KMeans(n_clusters=12)
gnc=GaussianNB()
clf=Pipeline([("imputer",imp),("scaler",mms),("pca",pca),("selection",kmc),("classifier",gnc)])

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.
my_dataset=data_dict
features_list=list(set(feature_list)-set(["poi"]))
features_list.insert(0,"poi")
dump_classifier_and_data(clf, my_dataset, features_list)
