'''
Contains methods that abstract natural language processing tasks
for joining NLP-generated features back into a dataframe and quickly
evaluating models.
'''
import numpy as np 
import pandas as pd 
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

def prep_nlp_df(X_og, X_new = False, transformer = None, drop_cols=[], text_col = 'text', new_columns=None):
    '''
    Abstracts the process of transforming the text column from a dataframe of features.
    If only original dataframe is provided, the text column is transformed and the resulting
    features are merged back into a dataframe. A transformer and name of the text column to 
    transform should be provided.
    If a second dataframe of previously transformed text features is provided, the transforming
    step is skipped and the dataframes are merged and returned without the specified columns.
    ---
    Arguments:
        X_og: Original dataframe to transform. Has column with text to transform.
        X_new: Dataframe of text features. Must have e
        drop_cols: list[str], names of columns to drop from final dataframe
        text_col: name of column in X_og that contains text strings to be transformed
    ---
    Returns:
        combo: Dataframe with transformed text features merged with original features.
               Unnecessary columns are dropped.
    '''
    # Check that input data is a pandas dataframe
    assert type(X_og) == type(pd.DataFrame())
    
    # Creating text features from the text column in the dataframe if no
    # features dataframe is provided
    if type(X_new) == bool:
        print('Making features from provided transformer and data...')
        sparse_new = transformer.transform(X_og[text_col])
        
        # Creating new columns.
        # If none are provided, use transformer's features.
        if not new_columns:
            new_columns = transformer.get_feature_names()
        
        #Create new features dataframe for merging
        X_new = pd.DataFrame(sparse_new.toarray(), columns = new_columns)
    
    #Merge Dataframes and drop unnecesary columns 
    assert type(X_new) == type(pd.DataFrame())
    assert np.shape(X_og)[0] == np.shape(X_new)[0]
    #combo_df = pd.merge(X_og, X_new, on=X_new.index)
    combo_df = pd.merge(X_og, X_new, right_index=True, left_index=True)
    
    return combo_df.drop(columns=drop_cols)

def lda_to_array(docs, n_topics, pd_format = False):
    '''
    Convert LDA document-topic representation into an array format.
    Each column in return array represents the contributions to all documents by that topic.
    ---
    Inputs:
        docs: list of documents transformed by LDA. Docs are represented by a list of tuples of
            form (int, float):
            [(<topic number>, <contribution to doc>)]
            Topic contributions do not appear for a document if less than 0.05,
            which is why this function is necessary.
        n_topics: int, number of topics used in LDA transformation
        pd_formatl: bool, returns array as Pandas DataFrame object if True
    Returns:
        document-topic matrix: Numpy or pandas array, each row represents a document,
            each column a topic
    '''
    
    # Pre-allocate all values to zero in case a document does not contain
    # a certain topic
    doc_top_mat = np.zeros((len(docs),n_topics))
    
    # For each topic-transformed doc, get the topic number and value
    # for topics represented in that document
    for row,doc in enumerate(docs):
        for topic in range(len(doc)):
            doc_top_mat[row,doc[topic][0]] = doc[topic][1]
            
    if pd_format:
        return pd.DataFrame(doc_top_mat,
                            columns =['topic{}'.format(i) for i in range(n_topics)])
    else:
        return doc_top_mat

def cross_val_and_test(X_train,y_train,X_test,y_test,model,cv=5,metric='accuracy'):
    '''
    Accepts training and test sets, and model instance with parameters for cross validation.
    Prints out cross-validation scores, test set scores, and returns trained model.
    ---
    Arguments:
        X_train,y_train,X_test,y_test: data for training and test sets. Object must be compatible
        with Scikit-Learn methods
        model: sklearn supervised classification algorithm object instance
        cv: integer or generator that produces indices for cross validation folds
        metric: string or callable, used to score the cross validation
    ---
    Returns:
        model: model instance trained on the training data set
    '''
    # Cross validate model on training set
    scores = cross_validate(model,X_train,y_train, cv=cv,scoring=metric, return_train_score=True) 
    print('Cross-Validation on training set:\n')
    print('Training score:  ', scores['train_score'].mean())
    print('Validation score:', scores['test_score'].mean())
    
    # Fit model to training set and predict on the test set
    model.fit(X_train,y_train)
    print('Test set results:\n')
    print(classification_report(y_test, model.predict(X_test)))
    
    return model