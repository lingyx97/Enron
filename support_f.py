def find_outliers(data_dict,var):
	"""find and return values and positions of outliers
	outliers are defined as those with value >= 2*(99 percentile) and
	those max values"""
	import numpy as np
	list_tmp=[]
	outliers=[]
	#list all the values of a feature
	for i in data_dict.keys():
		list_tmp.append(data_dict[i][var])#
	for i in data_dict.keys():
		#append outliers and their values to output list and ignore other values
		if np.abs(data_dict[i][var])>=2*np.nanpercentile(np.abs(list_tmp),95) or\
		np.abs(data_dict[i][var])==max(np.abs(list_tmp)):
			outliers.append((var,i,data_dict[i][var]))
	return outliers

def formatting_data(dat_dict,feat_list):
	"""generate feature and label sets from data dict and feature list.
	Then impute missing values and scale features."""
	import numpy as np
	from feature_format import featureFormat, targetFeatureSplit
	from sklearn.preprocessing import Imputer, MinMaxScaler
	my_dataset = dat_dict
	#make sure that "poi" is the first element of the list
	feat_list=list(set(feat_list)-set(["poi"]))
	feat_list=sorted(feat_list)
	feat_list.insert(0,"poi")
	# Extract features and labels from dataset for local testing
	data = featureFormat(my_dataset, feat_list, sort_keys=True)
	labels, features = targetFeatureSplit(data)
	labels,features=np.array(labels),np.array(features)

	#deal with missing values
	imp=Imputer(missing_values=np.nan,strategy="median")
	features=imp.fit_transform(features)

	#scaling features (problems can be caused by imputed missing values)
	mms=MinMaxScaler()
	features=mms.fit_transform(features)
	return features,labels

def find_best_PCA(features,labels,clf):
	"""iterate from 1 to 9 to decide the best n_component for given clf
	"""
	import numpy as np
	from sklearn.decomposition import PCA
	from sklearn.cross_validation import ShuffleSplit
	from sklearn.metrics import precision_score, recall_score
	PCA_score=[]
	for i in range(1,10):
		k=int(i)
		#define the PCA to be tested
		pca=PCA(n_components=k)
		features_tmp=pca.fit_transform(features)
		#generate training and testing sets
		ss= ShuffleSplit(len(features_tmp),10, test_size=0.5)
		scores=[] 	#accuracy scores (not used)
		preci=[]	#precision scores
		recall=[]	#recall scores
		for a,b in ss:
			feat_train,feat_test,lab_train,lab_test=features_tmp[a],features_tmp[b],\
				labels[a],labels[b]
			#score with given classifier
			clf.fit(feat_train,lab_train)
			# scores.append(clf.score(feat_test,lab_test))
			preci.append(precision_score(lab_test,clf.predict(feat_test)))
			recall.append(recall_score(lab_test,clf.predict(feat_test)))
		#find medians of each group
		PCA_score.append([np.median(preci),np.median(recall)])#,np.median(scores)])
	#return a list of indices where the best median presion and recall appears
	return list(np.argwhere( np.array(PCA_score)== np.amax(np.array(PCA_score),axis=0))[:,0])

def find_best_Kmeans(features,labels,clf):
	"""iterate from 1 to the largest number which guarantees an average of over
	ten points in a cluster.
	"""
	import math
	import numpy as np
	from sklearn.cluster import KMeans 
	from sklearn.cross_validation import ShuffleSplit
	from sklearn.metrics import precision_score, recall_score
	kmc_score=[]
	for i in range(1,math.floor(len(features)/10)):
		k=int(i)
		#define the Kmeans to be tested
		kmc=KMeans(n_clusters=k)
		features_tmp=kmc.fit_transform(features)
		#genetrate training and testing sets
		ss= ShuffleSplit(len(features_tmp),10, test_size=0.5)
		scores=[] 	#accuracy score (not used)
		preci=[]	#precision score
		recall=[]	#recall score
		for a,b in ss:
			feat_train,feat_test,lab_train,lab_test=features_tmp[a],features_tmp[b],\
				labels[a],labels[b]
			#score with given classifier
			clf.fit(feat_train,lab_train)
			# scores.append(clf.score(feat_test,lab_test))
			preci.append(precision_score(lab_test,clf.predict(feat_test)))
			recall.append(recall_score(lab_test,clf.predict(feat_test)))
		#find medians of each group
		kmc_score.append([np.median(preci),np.median(recall)])#,np.median(scores)])
	#return a list of indices where the best median presion and recall appears
	return list(np.argwhere( np.array(kmc_score)== np.amax(np.array(kmc_score),axis=0))[:,0])