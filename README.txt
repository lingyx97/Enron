Goal: Predict whether a person is Person Of Interest(POI) with given dataset.
Features: ['salary', 'deferral_payments', 
	'total_payments', 'loan_advances'(NOT USED), 
	'bonus', 'restricted_stock_deferred', 
	'deferred_income', 'total_stock_value', 
	'expenses', 'exercised_stock_options', 
	'other', 'long_term_incentive',
	 'restricted_stock', 'director_fees'] 
	(all units are in US dollars)
	 ['to_messages'(COMBINED), 'email_address'(NOT USED),
	'from_poi_to_this_person'(COMBINED), 'from_messages'(COMBINED), 
	'from_this_person_to_poi'(COMBINED), 'shared_receipt_with_poi'] 
	(units are generally number of emails messages)
Number of data points: 146, POI: 18, non-POI:128, ratio: 1-7.11 (very biased)

The dataset has high dimensionality, and the dependent variable (poi) is not 
	a result from direct combination of the features. i.e. we don't know 
	the relation between features and poi. Machine learning is thus useful.
	Also, as a result, all features were assumed to have equal weights in 
	prior.

There are a few outliers but only two were removed. One is TOTAL which is 
	obviously useless, and the other is LOCKHART EUGENE E who is a non-poi 
	with all other fields empty.THE TRAVEL AGENCY IN THE PARK is also a strange
	entry name, and thus has been removed. All other outliers were kept.

Only three people have non-empty loan_advances, and they have different poi 
	value, so the feature is removed.

from_poi_to_this_person and from_this_person_to_poi were conterted to percentage 
	of all emails to avoid bias because total emails a person send and 
	received does not have noticable relation with poi. deffered_income was 
	not combined with salary because both deffered_income and salary may carry 
	interesting information.Similar reasoning apply to payments.

The new variables contain leaked information from the future because we should not
	know who is poi before identify them (actually to_poi and from_poi themselves
	are problematic features).

director_fees, restricted_stock_deferred, deferral_payments, deferred_income, and
	long_term_incentive have over half empty values, and thus all empty values
	were filled with 0 to avoid trouble.

All NAN's in other features were replaced with the median of the existing values.

All features have been scaled into range [0,1] to avoid possible bias.

SelectKBest with scoring function f_classif is used to see which featuresdo the best in
	prediction. The following table is the mean score of 100 iterations.
'bonus', 		'deferral_payments', 		'deferred_income', 
'director_fees', 	'exercised_stock_options', 	'expenses', 
'long_term_incentive', 	'other', 			'pct_from_poi', 
'pct_to_poi', 		'restricted_stock',		'restricted_stock_deferred', 
'salary', 		'shared_receipt_with_poi', 	'total_payments', 
'total_stock_value'
15.8234102    		0.27000027  			10.25418475   
0.33380009  		27.1713641			1.00959622   
8.39079462   		3.90750558   			1.85730721  
15.85067103		8.38218804   			0.09723338  
10.88150104   		7.38569122   			8.3349912
23.43500494
Bonus, deferred_income, exercised_stock_options, pct_to_poi, saary, and total_stock_values
	have average scores larger than 10. High score of pct_to_poi may due to the
	information leakage described.

PCA is used to reduce dimensionality and then K-means to cluster candidates. 
	Arguments were decided by enumeration. Decusion tree was used at this stage
	but the arguments were not tuned because I wanted to reduce the tunning time
	by reducing loop number.

Ada-boost Classifier, Support Vector Machine, and Gaussian Naive Bayes were tried with
	the best combination of PCA and K-means found. Naive Bayes did the best in all
	cases while SVM without tuning gamma did the worst by predicting all people as 
	non-poi for most of the time.

Tunning parameters is to try different combinations of them and decide the best ones.The 
	reason I need to tune parameters is because Ada-boost Classifier and SVC both have 
	some arguments which can alter the behavior of classifiers. Those arguments were 
	decided by GridSearchCV according to F1 score.A bad argument could cause problems 
	like over-fitting (a decision tree with 200 leaves would always predict training 
	set with 100% accuracy but predict test set with very low accuracy) and 
	under-fitting (classify all candidates as non-poi).

Validation (train and test model on different subsets of features) can go wrong without 
	shuffling when the dataset is sorted by label, and thus I used shuffle cross 
	validation, and took the median F1 scores of all splitting combination.

(Median accuracy score,	Median f1 score,	Median f2 score)
(0.81944444444444442, 	0.41269841269841268, 	0.43174342105263153), 	Gaussian NB (component=1,cluster=12)
(0.85416666666666674, 	0.26737967914438499, 	0.21578947368421053), 	SVM (component=1,cluster=12)
(0.84722222222222221, 	0.25098039215686274, 	0.25062656641604009), 	Adaboost (component=1,cluster=12)
(0.84027777777777779, 	0.30769230769230765, 	0.2589285714285714), 	Gaussian NB (component=5,cluster=3)
(0.85416666666666674, 	0.2857142857142857, 	0.23458810692853246), 	SVM (component=5,cluster=3)
(0.79861111111111116, 	0.11805555555555558, 	0.11054421768707481)	Adaboost (component=5,cluster=3)

The accuracy is the percentage of correctly predicted labels out of all given labels.

The F1 score is the harmonic mean of precision and recall.

The F2 score is the weighted harmonic mean of precision(1x) and recall(2x).

The precision is the percentage of correctly predicted positive labels out of all
	positive predicted labels.

The recall is the percentage of correctly predicted positive labels out of all positive 
	given labels.

Final results given by tester.py:
	Accuracy: 0.84340
    	Precision: 0.40224
    	Recall: 0.35900
    	F1: 0.37939
    	F2: 0.36689
	Total predictions: 15000
    	True positives:  718
    	False positives: 1067
    	False negatives: 1282
    	True negatives: 11933

The accuracy is not the best because I tuned PCA and K-mean by precision and recall.

A recall of 0.357 is definitely too bad when trying to find out all possible poi's because
	65% of poi will be fugitives. This may due to the limitation of classifier itself 
	or missing features. Whichever the case, there is nothing I could do about it.
	Therefore I decided to submit the algorithm as it is.
