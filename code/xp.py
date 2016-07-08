#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
The file code/xp.py takes care of loading the data, fitting
the PLSR, and assessing the predictions using the Spearman rank-order correlation
coefficient. For a given experiment, it performs 10 runs, where each run are identical
except for the train/test split, which is random. The main parameters of the experiment
(data set, classifier, type of word vector, etc.) can be changed in the first lines of the
main function. 

To run the code, use python hw4xp.py

The code was tested with Python 2.7, SciPy 0.16.0 and scikit-learn between 0.16.1, and
NumPy 1.9.2. According to some deprecation warnings, it looks like scikit-learn 0.19
will raise some errors.

The AD, QMR, and GoogleNews-vectors-negative300 data sets
are provided in the /data folder.

2015-12-01
Franck Dernoncourt <francky@mit.edu>
'''


from __future__ import print_function
from __future__ import division
from distutils.version import LooseVersion, StrictVersion
import numpy as np
import random
import warnings
import cProfile
np.set_printoptions(precision=3, suppress=False, threshold=100000, linewidth=100000, edgeitems=10000)
np.random.seed(5)
import scipy.stats
import sklearn.cross_decomposition
import sklearn.linear_model  
import os
import sklearn.metrics.pairwise
#import mlpy # http://mlpy.sourceforge.net/docs/3.5/lin_regr.html#partial-least-squares

print('The NumPy version is {0}.'.format(np.version.version))
print('The scikit-learn version is {0}.'.format(sklearn.__version__))
print('The SciPy version is {0}\n'.format(scipy.version.full_version)) # requires SciPy >= 0.16.0

if LooseVersion(scipy.version.full_version) < '0.16.0': 
    raise ValueError('SciPy version should be >= 0.16.0')

def load_word2vec_vectors(vector_filepath, words_to_extract):
    '''
    Load word vectors in memory
    The EMNLP 2015 paper uses GoogleNews-vectors-negative300.zip
    Can be downloaded on https://code.google.com/p/word2vec/
    '''
        
    # Load GoogleNews-vectors-negative300: 
    # 1 row = 1 word + \t + its word vector (each float is separated by a comma)
    words2vec = {}
    vectors = np.loadtxt(vector_filepath, dtype=str, delimiter=' ')
    for vector in vectors:
        #print('vector: {0}'.format(vector))
        words2vec[vector[0]] = map(float, vector[1:-1])
    print('Word embedding dimension: {0}'.format(len(vector[1:-1])))
    return words2vec

def convert_quantifier_to_float(quantifier):
    '''
    A quantifier is a string, we convert it to a float
    '''
    map_annotation_numerical = {'ALL': 1, 'MOST': 0.95, 'SOME': 0.35, 'FEW': 0.05, 'NO': 0, 'CONCEPT': 0} # EMNLP 2015 paper, page 5, paragraph 2
    return map_annotation_numerical[quantifier.upper()] 

def convert_float_to_quantifier(my_float):
    '''
    Convert a float to its quantifier equivalent (string)
    '''
    map_numerical_quantifier = {1:'ALL', 0.95:'MOST', 0.35:'SOME', 0.05: 'FEW', 0:'NO'} # EMNLP 2015 paper, page 5, paragraph 2
    return map_numerical_quantifier[my_float] 

def load_semantic_space_data(data_folder, dataset_filename):
    '''
    Load the model-theoretic space, a.k.a. semantic space
    '''
    mcrae_quantified_majority_data_filepath = os.path.join(data_folder, dataset_filename)
    semantic_space_data = np.loadtxt(mcrae_quantified_majority_data_filepath, dtype=str)
    
    # Spearman's rank correlation coefficient
    # Reproduces: "Using the numerical data, we can calculate the mean 
    #             Spearman rank correlation between the three annotators,
    #             which comes to 0.63."
    print('Spearman correlation between annotators:')
    for annotator_1, annotator_2 in [(3,4), (4,5), (3,5)]:
        spearman = scipy.stats.spearmanr(map(convert_quantifier_to_float, semantic_space_data[:, annotator_1]), 
                                         map(convert_quantifier_to_float, semantic_space_data[:, annotator_2]))
        print('spearman: {0}'.format(spearman))


    # Create model-theoretic space
    concepts = np.unique(semantic_space_data[:, 1])
    number_of_concepts = len(concepts)
    features = np.unique(semantic_space_data[:, 2])
    number_of_features = len(features)
    print('\nnumber_of_concepts: {0}; number_of_features: {1}; number_of_samples: {2}'.
          format(number_of_concepts, number_of_features, semantic_space_data.shape[0]))
    semantic_space = scipy.sparse.lil_matrix((number_of_concepts, number_of_features), dtype=float) 
    semantic_space_defined = scipy.sparse.lil_matrix((number_of_concepts, number_of_features), dtype=np.int) # indicate whether a point the semantic space was defined in the data set
    
    # Keep track of annotations 
    annotations = {}
    for feature in features:
        annotations[feature] = {}
        for concept in concepts:
            annotations[feature][concept] = []
        
    # Populate model-theoretic space
    for semantic_space_sample in semantic_space_data:
        # Read sample
        concept = semantic_space_sample[1]
        feature = semantic_space_sample[2]
        quantifier = []
        quantifier.append(convert_quantifier_to_float(semantic_space_sample[3]))
        quantifier.append(convert_quantifier_to_float(semantic_space_sample[4]))
        quantifier.append(convert_quantifier_to_float(semantic_space_sample[5]))
        annotations[feature][concept].extend(quantifier)
        
        # Put sample in semantic space
        concept_index = np.where(concepts==concept)
        feature_index = np.where(features==feature)
        mean_quantifier = np.mean(quantifier)
        if mean_quantifier > 0: 
            semantic_space[concept_index, feature_index] = mean_quantifier 
        semantic_space_defined[concept_index, feature_index] = 1
    
    return semantic_space, semantic_space_defined, concepts, features, annotations


def assess_prediction_quality(y_true, y_pred):
    print('scipy.stats.describe(y_true_all): {0}'.format(scipy.stats.describe(y_true)))
    print('scipy.stats.describe(y_pred_all): {0}'.format(scipy.stats.describe(y_pred)))        
    spearman = scipy.stats.spearmanr(y_true, y_pred)
    print('spearman: {0}'.format(spearman))
    return spearman



def load_semantic_space(data_folder, semantic_space_data_set):
    if semantic_space_data_set == 'qmr':
        qmr, qmr_defined, concepts, features, annotations = load_semantic_space_data(data_folder, 'mcrae-quantified-majority_no_header.txt')
        training_set_size = 400 # Following the EMNLP 2015 paper train/test split percentage (Table 2)
    elif semantic_space_data_set == 'ad':
        qmr, qmr_defined, concepts, features, annotations = load_semantic_space_data(data_folder, 'herbelot_iwcs13_data_standard_format.txt')
        training_set_size = 60 # Following the EMNLP 2015 paper train/test split percentage (Table 2)
    else:
        raise ValueError('Invalid semantic_space_data_set')

    return qmr, qmr_defined, concepts, features, training_set_size, annotations


def load_word_vectors(word_vector_type, data_folder, concepts):
    if word_vector_type == 'random':
        words2vec = {}
        for concept in concepts:
            words2vec[concept] = np.random.random(300)
    else:
        vector_filepath = word_vector_type
        words2vec = load_word2vec_vectors(vector_filepath, concepts)  
    return words2vec
    
    
def main():
    '''
    ## Set experiment parameters
    data_folder = os.path.join('..', 'data')
    
    semantic_space_data_set = 'qmr'
    semantic_space_data_set = 'ad'
    
    word_vector_type = os.path.join(data_folder, 'GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms.txt')
    word_vector_type = os.path.join(data_folder, 'GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms-plus.txt')
    word_vector_type = os.path.join(data_folder, 'GoogleNews-vectors-negative300-short-onlyspaces.txt')
    word_vector_type = os.path.join(data_folder, 'GoogleNews-vectors-negative300-short-retrofit-ppdb.txt')
    word_vector_type = os.path.join(data_folder, 'GoogleNews-vectors-negative300-short-retrofit-framenet.txt')
    word_vector_type = 'random' 
    
    model_type = 'LR'
    model_type = 'true-mode'
    model_type = 'mode'
    model_type = 'PLSR'
    model_type = 'nearest-neighbor-in-train-set' 
    '''
    
    
    
    # python hw4xp_solutions_release.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-onlyspaces.txt > 
    # python hw4xp_solutions_release.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-framenet.txt
    # python hw4xp_solutions_release.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-ppdb.txt
    # python hw4xp_solutions_release.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms.txt
    # python hw4xp_solutions_release.py -m PLSR -s ad -v ../data/GoogleNews-vectors-negative300-short-retrofit-wordnet-synonyms-plus.txt
    # python hw4xp_solutions_release.py -m PLSR -s ad -v random > ad_random_plsr.txt
    # python hw4xp_solutions_release.py -m nn -s ad -v random > ad_random_nn.txt
    # python hw4xp_solutions_release.py -m nn -s ad -v word2vec > ad_word2vec_nn.txt
    # python hw4xp_solutions_release.py -m mode -s ad  > ad_mode.txt
    # python hw4xp_solutions_release.py -m true-mode -s ad > ad_true-mode.txt
    
    # Thanks Tej Chajed for the command line interface :) https://piazza.com/class/iedgnb0z8r89y?cid=230
    import argparse

    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--semantic-dataset',
            dest='semantic_data_set',
            choices=['qmr', 'ad'],
            default='ad',
            help='semantic space data set')
    parser.add_argument('-w', '--wordvec',
            choices=['random', 'word2vec'],
            default='word2vec',
            help='type of word vectors')
    parser.add_argument('-m', '--model',
            choices=['nn', 'mode', 'true-mode', 'LR', 'PLSR'],
            default='PLSR',
            help='model to use for training')
    # the default data directory is determined relative to this script
    script_path = os.path.dirname(__file__)
    data_path = os.path.join(script_path, '..', 'data')
    parser.add_argument('-d', '--data',
            metavar='DATA_DIR',
            default=os.path.normpath(data_path),
            help='data directory to load word vectors and semantic datasets from')
    parser.add_argument('-v', '--word-vectors',
            dest='word2vec_filename',
            default='../data/GoogleNews-vectors-negative300-short-onlyspaces.txt',
            help='name of file in DATA_DIR with word vectors')

    args = parser.parse_args()

    ## Set experiment parameters

    semantic_space_data_set = args.semantic_data_set
    word_vector_type = args.word2vec_filename
    model_type = args.model
    if model_type == 'nn':
        model_type = 'nearest-neighbor-in-train-set'
    
    # Misc parameters
    data_folder = os.path.join('..', 'data')#args.data
    
    
    ## Load data
    semantic_space, semantic_space_defined, concepts, features, training_set_size, annotations = load_semantic_space(data_folder, semantic_space_data_set)
    word_embeddings = load_word_vectors(word_vector_type, data_folder, concepts)
    print('semantic_space.shape: {0}'.format(semantic_space.shape))


    ## PLS-regression
    #Prepare data        
    X = []
    y_true = []
    y_true_defined = []
    concepts_present = []
    for concept_id, concept in enumerate(concepts):
        if concept in word_embeddings:
            X.append(word_embeddings[concept])
            y_true.append(semantic_space[concept_id, :].todense().tolist()[0])
            y_true_defined.append(semantic_space_defined[concept_id, :].todense().tolist()[0])            
            concepts_present.append(concept)
    
    X = np.array(X)
    y_true = np.array(y_true)
    y_true_defined = np.array(y_true_defined)
    concepts_present = np.array(concepts_present)
    
    # Perform several runs so that we get some statistically robust results
    run_number = 0
    total_run_number = 1000
    results = {}
    results['spearmans'] = {}
    results['spearmans']['only_annotated'] = {}
    results['spearmans']['only_annotated']['SpearmanrResult'] = []
    results['spearmans']['only_annotated']['correlation'] = []
    results['train_set_idx'] = []
    results['test_set_idx'] = []
    
    while run_number < total_run_number:
    
        print('\n\n##### Run {0} out of {1} #####'.format(run_number+1, total_run_number))
        # Instantiate the model
        if model_type == 'PLSR':
            pls2 = sklearn.cross_decomposition.PLSRegression(n_components=20, max_iter=5000, scale=False, tol=1e-06)
        elif model_type == 'LR':
            pls2 = sklearn.linear_model.LinearRegression()        
        elif model_type in ['average', 'mode', 'nearest-neighbor-in-train-set', 'true-mode']:
            pass
        else:
            raise ValueError('model_type is invalid')
        
        # Split train/test
        train_set_idx = np.array(random.sample(range(X.shape[0]), training_set_size))
        test_set_idx =  np.setdiff1d(range(0,X.shape[0]), train_set_idx)
        assert(np.intersect1d(train_set_idx, test_set_idx).shape[0]==0) # Ensure that the test set is not contaminated with training samples
        
        y_train_true = y_true[train_set_idx, :]
        y_train_true_defined = y_true_defined[train_set_idx, :]
        print('concepts_present.shape: {0}'.format(concepts_present.shape))
        concepts_present_train = concepts_present[train_set_idx]
        concepts_present_test = concepts_present[test_set_idx]
        
        # Learn the model
        if model_type in ['LR', 'PLSR']:
            xx = X[train_set_idx, :]
            yy = y_true[train_set_idx, :]                        
            # Try-except to bypass sklearn's bug: https://github.com/scikit-learn/scikit-learn/issues/2089#issuecomment-152753095
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pls2.fit(xx, yy)
            except:
                print('sklearn.cross_decomposition.PLSRegression() crashed due to scikit-learn\'s bug https://github.com/scikit-learn/scikit-learn/issues/2089#issuecomment-152753095. Re-doing the run with a new train/test split.')
                continue
            
        elif model_type in ['PLSR-mlpy']:
            pls2.learn(xx, yy)
        elif model_type in ['mode']:
            mode_y_train_true = np.zeros(y_true.shape[1])
            for feature_number in range(len(features)):
                feature_values = []
                for concept_number in range(y_train_true.shape[0]): 
                    if y_train_true_defined[concept_number, feature_number] == 1: 
                        feature_values.append(y_train_true[concept_number, feature_number])
                if len(feature_values) == 0:
                    mode_y_train_true[feature_number] = 0
                else:
                    mode_y_train_true[feature_number] = scipy.stats.mode(feature_values).mode
            
        elif model_type in ['true-mode']:
            mode_y_train_true = []
            train_concepts = np.array(concepts_present)[train_set_idx]
            
            feature_mode = {}
            for feature in features:
                feature_values = []
                for concept in train_concepts: # make sure we don't use the test set
                    feature_values.extend(annotations[feature][concept])
                #print('feature: {0}\tfeature_values: {1}'.format(feature, feature_values))
                if len(feature_values) > 0:
                    feature_mode[feature] = scipy.stats.mode(feature_values).mode.tolist()
                else:
                    feature_mode[feature] = [0]
                mode_y_train_true.extend(feature_mode[feature])
            
            
            
        elif model_type in ['nearest-neighbor-in-train-set']:
            pass
        
        # Use the learnt model to make predictions
        if model_type in ['LR', 'PLSR']:        
            y_pred = pls2.predict(X[test_set_idx, :])
        elif model_type in ['PLSR-mlpy']:
            pls2.pref(X[test_set_idx, :])
        elif model_type in ['mode']:
            y_pred = np.tile(mode_y_train_true, (X[test_set_idx, :].shape[0], 1))
        elif model_type in ['true-mode']:
            y_pred = np.tile(mode_y_train_true, (X[test_set_idx, :].shape[0], 1))
        elif model_type in ['nearest-neighbor-in-train-set']:
            
            y_pred = np.zeros((X[test_set_idx, :].shape[0], y_train_true.shape[1]))
            for test_concept_number, test_concept_word_vector in enumerate(X[test_set_idx, :]):
                # Find most similar concept in the training set. Similarity is based on the cosine similarity of the word vectors
                max_cosine_similarity = -999
                best_prediction = ''
                closest_concept = -1
                for train_concept_number, train_concept_word_vector in enumerate(X[train_set_idx, :]):
                    cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(test_concept_word_vector, train_concept_word_vector)
                    if cosine_similarity > max_cosine_similarity:
                        max_cosine_similarity = cosine_similarity
                        best_prediction = y_train_true[train_concept_number, :]
                        closest_concept = concepts_present_train[train_concept_number]
                y_pred[test_concept_number, :] = best_prediction
                # Note: to get the NN, one could have used:
                # - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
                # - http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_argmin_min.html
                
                
        # Prepare test set
        y_test_true = y_true[test_set_idx, :]
        y_test_true_defined = y_true_defined[test_set_idx, :]
        
        y_pred_all = y_pred.flatten()
        y_test_true_all = y_test_true.flatten()
        y_test_true_defined_all = y_test_true_defined.flatten()
        
        # Assess model removing the features that are not specified in the human annotations, for each concept
        # (a concept has an average 11 annotated features)
        y_pred_all_only_annotated = []
        y_true_all_only_annotated = []
        for y_t, y_p, y_t_d in zip(y_test_true_all, y_pred_all, y_test_true_defined_all):
            if y_t_d == 1:
                y_pred_all_only_annotated.append(y_p)
                y_true_all_only_annotated.append(y_t)
        spearman = assess_prediction_quality(y_true_all_only_annotated, y_pred_all_only_annotated)
        
        # Save results
        results['test_set_idx'].append(test_set_idx)
        results['train_set_idx'].append(train_set_idx)
        results['spearmans']['only_annotated']['SpearmanrResult'].append(spearman)
        results['spearmans']['only_annotated']['correlation'].append(spearman.correlation)        

        run_number += 1
        
    ## Display results
    print('\n\n\n##### Result summary for the {0} runs #####'.format(total_run_number))
    for run_number in range(len(results['train_set_idx'])):
        print('Run {3:02d}: \tspearman.correlation: {0}\t train_set_idx: {1}\t test_set_idx: {2}'.format(results['spearmans']['only_annotated']['correlation'][run_number],
                                                 results['train_set_idx'][run_number],
                                                 results['test_set_idx'][run_number],
                                                 run_number+1))
    
    print('\nCorrelation stats over {1} runs (below are only numbers you should compare with the paper and report in the homework):\n{0}'.format(scipy.stats.describe(results['spearmans']['only_annotated']['correlation']), total_run_number))
    print('semantic_space_data_set: {0}'.format(semantic_space_data_set))
    print('word_vector_type: {0}'.format(word_vector_type))
    print('model_type: {0}'.format(model_type))
    print('Solutions')
    
if __name__ == "__main__":
    main()
    #cProfile.run('main()') # if you want to do some profiling