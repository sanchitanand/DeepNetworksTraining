import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from decimal import Decimal


#files
x_train_file_name = 'train_X.npy'
y_train_file_name = 'train_binary_Y.npy'
x_test_file_name = "valid_test_X.npy"
output_file_name = "valid_test_Y.npy"
log_file_name = "mylogfile.txt"

#hyperparameters
n_hidden_layer_neurons = 1000
batch_size = 200
num_epochs = 300
regularization_const = tf.constant(5e-4)
dropout_const = 0.7368063
adam_const = 1e-3

def load_training():
	x_train = np.load(x_train_file_name)
	x_train = x_train.reshape((x_train.shape[0],-1))
	y_train = np.load(y_train_file_name)
	return (x_train,y_train)
	
def load_test():
	x_test = np.load(x_test_file_name.npy)
	x_test = x_test.reshape((x_test.shape[0],-1))
	return x_test
	
def log_debug_data(epoch,f,test_acc_temp,temp_loss,train_acc_temp,print_output):
	print("epoch:          %-10d" % epoch,file = f)
	print('train_loss:     %-10.3E' % Decimal(temp_loss.item()),file = f)
	print("train_accuracy: %-10s" %str(np.round(train_acc_temp,3)),file = f)
	print("test_accuracy:  %-10s" %str(np.round(test_acc_temp,3)), file = f)
	
	if print_output:
		print("epoch:          %-10d" % epoch)
		print('train_loss:     %-10.3E' % Decimal(temp_loss.item()))
		print("train_accuracy: %-10s" %str(np.round(train_acc_temp,3)))
		print("test_accuracy:  %-10s" %str(np.round(test_acc_temp,3)))
	
	
if __name__ == '__main__':
	f = open(log_file_name,"w")
	x,y = load_training() 
	testing_data = load_test()
	
	n_features = x.shape[1]
	n_classes = y.shape[1]
	
	
	bound_in = (6/(n_features + n_hidden_layer_neurons))**0.5
	bound_hidden = (6/(2*n_hidden_layer_neurons))**0.5
	bound_out = (6/(19 + n_hidden_layer_neurons))**0.5
	sess = tf.Session()
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
	#x_train = x   #use this for submitting results to competition
	#y_train = y   #use this for submitting results to competition		
	X = tf.placeholder(dtype=tf.float32, shape=[None, n_features])
	Y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes ])
	keep_prob = tf.placeholder(tf.float32)
	
	W_hidden_1 = tf.Variable(tf.random_normal([n_features,n_hidden_layer_neurons])*bound_in)
	B_hidden_1 = tf.Variable(tf.random_normal([n_hidden_layer_neurons])*bound_in)
	W_hidden_2 = tf.Variable(tf.random_normal([n_hidden_layer_neurons,n_hidden_layer_neurons])*bound_hidden)
	B_hidden_2 = tf.Variable(tf.random_normal([n_hidden_layer_neurons])*bound_hidden)
	W_hidden_3 = tf.Variable(tf.random_normal([n_hidden_layer_neurons,n_hidden_layer_neurons])*bound_hidden)
	B_hidden_3 = tf.Variable(tf.random_normal([n_hidden_layer_neurons])*bound_hidden)
	W_hidden_4 = tf.Variable(tf.random_normal([n_hidden_layer_neurons,n_hidden_layer_neurons])*bound_hidden)
	B_hidden_4 = tf.Variable(tf.random_normal([n_hidden_layer_neurons])*bound_hidden)
	W_hidden_5 = tf.Variable(tf.random_normal([n_hidden_layer_neurons,n_hidden_layer_neurons])*bound_hidden)
	B_hidden_5 = tf.Variable(tf.random_normal([n_hidden_layer_neurons])*bound_hidden)
	W_hidden_6 = tf.Variable(tf.random_normal([n_hidden_layer_neurons,n_hidden_layer_neurons])*bound_hidden)
	B_hidden_6 = tf.Variable(tf.random_normal([n_hidden_layer_neurons])*bound_hidden)
	W_out = tf.Variable(tf.random_normal([n_hidden_layer_neurons,n_classes])*bound_out)
	B_out = tf.Variable(tf.random_normal([n_classes])*bound_out)
	
	
	X_norm = tf.nn.batch_normalization(X,0,1,0,1.,1e-8)
	
	hidden_0 = tf.nn.relu(tf.add(tf.matmul(X_norm, W_hidden_1), B_hidden_1))
	hidden_0 = tf.nn.dropout(hidden_0,keep_prob)
	hidden_1 = tf.nn.relu(tf.add(tf.matmul(hidden_0, W_hidden_2), B_hidden_2))
	hidden_1 = tf.nn.dropout(hidden_1,keep_prob)
	hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_3), B_hidden_3))
	hidden_2 = tf.nn.dropout(hidden_2,keep_prob)
	hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_4), B_hidden_4))
	hidden_3 = tf.nn.dropout(hidden_3,keep_prob)
	hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_5), B_hidden_5))
	hidden_4 = tf.nn.dropout(hidden_4,keep_prob)
	hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_6), B_hidden_6))
	hidden_5 = tf.nn.dropout(hidden_5,keep_prob)
	out1 = tf.add(tf.matmul(hidden_5,W_out),B_out)
	
	out2 = tf.nn.sigmoid(out1)
	
	w = 1/7
	L2_norm = w*tf.reduce_mean(tf.square(W_out)) + w*tf.reduce_mean(tf.square(W_hidden_1)) + w*tf.reduce_mean(tf.square(W_hidden_2)) + w*tf.reduce_mean(tf.square(W_hidden_3))+ w*tf.reduce_mean(tf.square(W_hidden_4))+ w*tf.reduce_mean(tf.square(W_hidden_5))+ w*tf.reduce_mean(tf.square(W_hidden_6))
	pred = tf.cast(tf.greater(out2,0.5),tf.float32)


	loss = tf.losses.sigmoid_cross_entropy(Y,out1, reduction = tf.losses.Reduction.NONE)
	loss = (1 - regularization_const) * tf.reduce_mean(loss) + regularization_const * L2_norm
	
	subset = tf.equal(1.,tf.reduce_mean(tf.cast(tf.equal(pred,Y),tf.float32),1))
	acc =  tf.reduce_mean(tf.cast(subset,tf.float32))
	opt = tf.train.AdamOptimizer(adam_const)
	train_step = opt.minimize(loss)
	
	init = tf.global_variables_initializer()
	sess.run(init)
	saver = tf.train.Saver()
	
	for i in range(num_epochs):
		x_train,y_train = shuffle(x_train,y_train)
		for j in range(x_train.shape[0]//batch_size):
			rand_x = x_train[batch_size*j : batch_size*(j+1)]
			rand_y = y_train[batch_size*j : batch_size*(j+1)]
			sess.run(train_step,feed_dict={X : rand_x, Y : rand_y, keep_prob :dropout_const})
		
		acc_test = sess.run(acc, feed_dict = {X : x_test, Y : y_test, keep_prob : 1.0}) #comment these out for competition code
		loss_t = sess.run(loss,feed_dict = {X : rand_x, Y : rand_y, keep_prob : 1.0}) #comment these out for competition code
		acc_train = sess.run(acc, feed_dict = {X : rand_x, Y : rand_y, keep_prob : 1.0}) #comment these out for competition code
		log_debug_data(i,f,acc_test,loss_t,acc_train,True) #comment these out for competition code

	testing_predictions = sess.run(pred,feed_dict = {X : testing_data, keep_prob : 1.0})
	np.save(output_file_name,testing_predictions)
	f.close()