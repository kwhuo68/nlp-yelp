import os
import math
import collections
import numpy as np
import tensorflow as tf


class Word2Vec():
	def __init__(self):
		self.learning_rate = 0.001
		self.word_index = 0
		self.vocab_size = 5000
		self.embedding_size = 300
		self.batch_size = 128
		self.window_size = 2
		self.setup()
		self.session = tf.Session(graph = self.graph)


	#Build relevant lists and dictionaries given list of tokens
	def build_dataset(self, words, vocab_size = 5000):
		#Keep track of counts of "rare" words
		counts = [['unknown', -1]]
		counts.extend(collections.Counter(words).most_common(vocab_size - 1))
		word_index_map = dict()
		word_indices = list()
		unknown_count = 0

		#Get counts, and unknown count (index 0)
		for word, num in counts:
			word_index_map[word] = len(word_index_map)
		for word in words:
			if word in word_index_map:
				index = word_index_map[word]
			else:
				index = 0 #unknown
				unknown_count += 1
			word_indices.append(index)
		counts[0][1] = unknown_count 
		reverse_word_index_map = dict(zip(word_index_map.values(), word_index_map.keys()))
		return word_indices, counts, word_index_map, reverse_word_index_map 

	#Get batch & label pairs
	def get_batches(self, training_words, batch_size, window_size):
		context_window = 2 * self.window_size #context words of length window before and after target
		batch = np.ndarray(shape = (batch_size, context_window), dtype = np.int32)
		labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)
		window_span = context_window + 1 #includes target 
		
		sliding_window = collections.deque(maxlen = window_span) #[window, target, window]
		for i in range(window_span):
			sliding_window.append(training_words[self.word_index])
			self.word_index = (self.word_index + 1) % len(training_words)

		#Process in windows
		for j in range(batch_size):
			batch[j, :] = [word for index, word in enumerate(sliding_window) if index != self.window_size] #target is at index window_size
			labels[j, 0] = sliding_window[self.window_size]
			sliding_window.append(training_words[self.word_index])
			self.word_index = (self.word_index + 1) % len(training_words)

		return batch, labels

	#Setup tf graph
	def setup(self, window_size = 5, learning_rate = 0.01, epochs = 10, batch_size = 50):
		print("Setting up tensorflow graph..")
		loss = 0 
		loss_changes = []

		self.graph = tf.Graph()

		with self.graph.as_default():
			#inputs and labels
			self.train_inputs = tf.placeholder(tf.int32, shape = [self.batch_size, self.window_size * 2]) #CBOW implementation 
			self.train_labels = tf.placeholder(tf.int32, shape = [self.batch_size, 1])

			#embeddings
			self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0))
			#self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
			embed = tf.zeros([self.batch_size, self.embedding_size])
			for i in range(self.window_size * 2):
				embed += tf.nn.embedding_lookup(self.embeddings, self.train_inputs[:, i])
			self.embed = embed

			#set softmax weights, biases, output 
			self.weights = tf.Variable(tf.truncated_normal(shape = [self.vocab_size, self.embedding_size], mean = 0.0, stddev = 1.0/(math.sqrt(self.embedding_size))))
			self.biases = tf.Variable(tf.zeros([self.vocab_size]))
			self.output = tf.matmul(self.embed, tf.transpose(self.weights)) + self.biases 

			#loss
			self.train_labels_one_hot = tf.one_hot(self.train_labels, self.vocab_size)
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.train_labels_one_hot)) #cross entropy

			#optimizer
			self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

			#embeddings
			embed_norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims = True))
			self.normalized_embeddings = self.embeddings / embed_norm

			#initialize and save
			self.init_op = tf.initialize_all_variables()
			self.saver = tf.train.Saver()

			print("Done with initial tensorflow setup..")


	#Run using above setup, keeping track of losses
	def run(self, words, iterations):
		word_indices, counts, word_index_map, reverse_word_index_map = self.build_dataset(words, self.vocab_size)
		self.word_indices = word_indices
		self.counts = counts
		self.word_index_map = word_index_map
		self.reverse_word_index_map = reverse_word_index_map

		#Session initialization 
		session = self.session
		session.run(self.init_op)
		avg_loss = 0

		#Calculate losses and print
		for i in range(iterations):
			batch, labels = self.get_batches(self.word_indices, self.batch_size, self.window_size)
			feed = {self.train_inputs: batch, self.train_labels: labels}
			op, loss = session.run([self.optimizer, self.loss], feed_dict = feed)
			avg_loss += loss
			if i % 100 == 0:
				if i > 0:
					avg_loss /= 100
					print("Average loss by iteration %d : %f" % (i, avg_loss))
					avg_loss = 0
		self.final_embeddings = self.normalized_embeddings.eval(session = self.session)

		return self






