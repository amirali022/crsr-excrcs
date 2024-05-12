### Coursera - DeepLearning.AI TensorFlow Developer Professional Assignments

1. introduction tensorflow
	- a new programming paradigm
		- [Introduction](/house_price_prediction.ipynb)
			- simple Hello World example using tensorflow

	- introduction to computer vision
		- [Introduction to Computer Vision](/fashion_mnist.ipynb)
			- simple example of classifying fashion mnist dataset using multi layer neural network
			- working with callback
		- [Weekly Assignment](/mnist_mlp.ipynb)
			- classify mnist dataset using multi layer neural network
			- plus self made early stopping callback
	- enhancing vision with convolutional neural networks
		- [Fashion MNIST Classification Using CNN](/fashion_convolution.ipynb)
			- classify fashion mnsit using convolution neural networks
			- visualizing filters
		- [Exploring Convolutions & Max Pooling](/convolutions.ipynb)
			- Implement convolution with 3x3 filters
			- Implement 2x2 Max Pooling Filter
		- [Weekly Assignment](/fashion_convolution.ipynb)
			- classify mnist dataset using CNN
			- plus self made early stopping callback
	- using real world images
		-[Horses or Humans Classfication Using CNN](/horse_or_human.ipynb)
			- download and prepare dateset using image generator
			- train and validation set
			- visualizing layers output of model
			- test real images

2. convolutional neural networks
	- exploring a larger dataset
		- [Cats vs. Dogs](/cats_vs_dogs.ipynb)
	- augmentation a technique to avoid overfitting
		- [Cats vs. Dogs Augmentation](/cats_vs_dogs_with_augmentation.ipynb)
			- adding augmentation preprocesses such as rotation, width_shift, height_shift, shear, zoom, horizontal_flip
		- [Horse or Human Augmentation](/horse_or_human_with_augmentation.ipynb)
			- adding augmentation (however does not have impressive impact on validation accuracy)
	- transfer learning
		- [Cats vs. Dogs Transfer Learning](/transfer_learning.ipynb)
			- using pre-trained inception model on imagenet for cats vs dogs classification
	- multi-class classification
		- [Rock Paper Scissors Classification](/rock_paper_scissors_cnn.ipynb)
			- using cnn to classify images of rps dataset

3. natural language processing
	- sentiment in text
		- [Text Preprocessing](/text_processing.ipynb)
			- work with tokenizer and pad sequences
			- work with sarcasm dataset
	- word embedding
		- [IMDB Reviews Sentiment Analysis](/imdb_reviews.ipynb)
			- working with embedding layer
			- visualizing embedding weights
		- [Sarcasm Classification](/sarcasm_classifier.ipynb)
			- binary classification problem of sarcasm detection
		- [IMDB Reviews Classification Using Subwords](/imdb_reviews_subwords.ipynb)
			- using imdb_reviews/subwords8k dataset
			- comparing plaintext dataset with subwords dataset
			- comparing texts_to_sequences/sequences_to_texts with encode/decode
	- sequence models
		- [IMDB Reviews Subwords Classification Using LSTM](/imdb_reviews_subwords_lstm.ipynb)
			- classification using two layer bidirectional lstm network
		- [IMDB Reviews Subwords Classification Using CNN](/imdb_reviews_subwords_cnn.ipynb)
			- classification using Conv1D layer
	- sequence models and literature
		- [Text Generation](/generate_texts.ipynb)
			- working with Bidirectional LSTM inorder to generate text
		- [Assignment](/Copy_of_C3W4_Assignment.ipynb)
			- generate text

4. sequences time series and prediction
	- sequences and prediction
		- [Time Series](/time_series.ipynb)
			- introduction to time series
		- [Forecast](/forecast.ipynb)
			- forecast synthetic data using statistical methods
	- deep neural networks for time series
		- [Time Series Data](/time_series_feature_labels.ipynb)
			- preparing dataset of time series
			- windowed data
			- feature and labels
			- batch data
			- shuffle data
		- [TS Prediction Using Single Layer Neural Network](/time_series_prediction_single_layer.ipynb)
			- prediction of synthetic data
			- printing weights of neuron
		- [TS Prediction Using Multi Layer Neural Netword](/time_series_prediction_multi_layer.ipynb)
			- prediction of synthetic data
			- tuning learning rate
		- [TS Prediction Using RNN](/time_series_prediction_rnn.ipynb)
			- prediction of synthetic data using recurrent network
			- tuning learning rate
		- [TS Prediction Using LSTM](/time_series_prediction_lstm.ipynb)