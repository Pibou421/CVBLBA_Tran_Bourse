# Implementation of Collapsed Variational Bayesian Inference for Latent Dirichlet Allocation

This Github aims at implementing a version of the Collapsed Variational Bayesian Inference for Latent Dirichlet Allocation algorithm described by the paper written by Teh Y.W., Newman D., and Welling, M.
in Advances in Neural Information Processing Systems (NIPS) 19, 2007.

This code is written in C and Python with help from the module Cython in order to test the method to its limits especially for the convergence speed. Unfortunately this code still needs debugging as the module cython had some issues with our libraries like numpy.

## Run the program:

You can run the program thanks to the command: 

'python cvb_LDA.py $text_file_path $language $nb_topics $nb_words_per_topic ${optional}comparison ${optional}alpha ${optional}beta ${optional}max_iter ${optional}tolerance'

The non optionnal parameters are :

* 'text_file_path' (str): path to the file containing the documents to analyze (format: one document per line, exactly one whitespace between each token)
* 'language' (str):used to process input text (remove stopwords): can be 'french' or 'english'
* 'nb_topics' (int): number of topics the LDA algorithm will consider
* 'nb_words_per_topic' (int): the number of words which will be output linked to each topic
* {optional}comparison (bool): comparison with the Sklearn LDA algorithm ? (default= 'true') or with other like collapsed Gibbs sampling if implemented.

The other optionnal parameters are self explicit within the paper and with the notations.

## Output format:

Output the nb_words_per_topic most representative words of each topic (the number of different topics being equal to nb_topics).
