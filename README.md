# STAT 946 - Data Challenge 2 - Sentence Ordering

For this data challenge, we create models to reorder a given set of sentences from Wikipedia.

We take a two-step approach:

1. We use a BERT sequence classifier to predict the first sentence in the given set of sentences. We recognize that the first sentence usually has distinct characteristics and patterns (e.g., Person-Date-Date)
2. We use another BERT sequence classifier to predict pairwise orders between two sentences - if sentence A appears before sentence B, then the label is 0, otherwise 1. The order does NOT have to be consecutive
Using the results from the two classifers, we can construct a directed graph - each sentence forms a node and the logits from the pairwise classifer become the edge weights. To form a prediction, we simply search for the path (starting from the predicted first sentence/node) that maximizes the total pairwise edge weights.

We randomly select 10,000/500 documents from the labeled dataset as our training/validation set respectively.

We use WeightedRandomSampler with a target ratio of 0.5 for the first sentence classifier to accommodate the class imbalance, and simple RandomSampler for the pairwise classifier since the class ratio is already balanced.

For model selection, we use validation accuracy for the first sentence classifier and validation mean column-wise Spearman correlation for the pairwise classifier.
