# nlp-yelp

Looking at whether word embeddings can be used to predict how "useful" reviews on Yelp are. Word embeddings are generated through a word2vec approach (using a Continuous Bag-of-Words implementation) over reviews via tensorfow. These vectors are then aggregated per review and used as features for logistic regression (optimized via gradient descent). End classification output (useful or not) is compared to baseline.  
