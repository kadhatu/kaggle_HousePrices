#!/usr/bin/python

import pandas as pd 
import numpy as np 
from pandas import Series, DataFrame
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing


# import datasets
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")


# Pretreatment function
def set_missing_type(df, feature_name):
    df.loc[ (df[feature_name].isnull()), feature_name ] = "NA"
    return

def set_dummies(df, feature_name):
    return pd.get_dummies(df[feature_name], prefix= feature_name)
    
def feature_scaler(df, feature_name):
    scaler = preprocessing.StandardScaler()
    tmp_scale_param = scaler.fit(df[feature_name].values.reshape(-1,1))
    df[feature_name] = scaler.fit_transform(df[feature_name].values.reshape(-1,1), tmp_scale_param)
    return


# set one-hot encode for no-nonnumeric features
MSZoning_dummies = set_dummies(data_train, 'MSZoning')
Street_dummies = set_dummies(data_train, 'Street')
LotShape_dummies = set_dummies(data_train, 'LotShape')
LandContour_dummies = set_dummies(data_train, 'LandContour')
LotConfig_dummies = set_dummies(data_train, 'LotConfig')
LandSlope_dummies = set_dummies(data_train, 'LandSlope')
Neighborhood_dummies = set_dummies(data_train, 'Neighborhood')

MSZoning_dummies_test = set_dummies(data_test, 'MSZoning')
Street_dummies_test = set_dummies(data_test, 'Street')
LotShape_dummies_test = set_dummies(data_test, 'LotShape')
LandContour_dummies_test = set_dummies(data_test, 'LandContour')
LotConfig_dummies_test = set_dummies(data_test, 'LotConfig')
LandSlope_dummies_test = set_dummies(data_test, 'LandSlope')
Neighborhood_dummies_test = set_dummies(data_test, 'Neighborhood')


# datasets re-construction to the copies of df
df = pd.concat([data_train, MSZoning_dummies, Street_dummies, LotShape_dummies, LandContour_dummies, LotConfig_dummies, LandSlope_dummies, Neighborhood_dummies], axis=1)
df_test = pd.concat([data_test, MSZoning_dummies_test, Street_dummies_test, LotShape_dummies_test, LandContour_dummies_test, LotConfig_dummies_test, LandSlope_dummies_test, Neighborhood_dummies_test], axis=1)


# feature scale for df 
feature_scaler(df, 'MSSubClass')
feature_scaler(df, 'LotArea')
feature_scaler(df, 'OverallQual')
feature_scaler(df, 'OverallCond')
feature_scaler(df, 'YearBuilt')
feature_scaler(df, 'YearRemodAdd')
feature_scaler(df_test, 'MSSubClass')
feature_scaler(df_test, 'LotArea')
feature_scaler(df_test, 'OverallQual')
feature_scaler(df_test, 'OverallCond')
feature_scaler(df_test, 'YearBuilt')
feature_scaler(df_test, 'YearRemodAdd')


# df features filter 
df.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood'], axis=1, inplace=True)
train_df = df.filter(regex="MSZoning_.*|Street_.*|MSSubClass|LotArea|OverallQual|OverallCond|YearBuilt|YearRemodAdd|LotShape_.*|LandContour_.*|LotConfig_*|LandSlope_*|Neighborhood|SalePrice")
df_test.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood'], axis=1, inplace=True)
test_df = df_test.filter(regex="MSZoning_.*|Street_.*|MSSubClass|LotArea|OverallQual|OverallCond|YearBuilt|YearRemodAdd|LotShape_.*|LandContour_.*|LotConfig_*|LandSlope_*|Neighborhood")


# set matrix for train and test
train_np = train_df.values
train_np[:, [0, 6]] = train_np[:, [6, 0]]
test_np = test_df.values
y_train = train_np[:, 0, np.newaxis]
x_train = train_np[:, 1:]
x_test = test_np[:, 0:]
ymin, ymax = y_train.min(), y_train.max() 
y_std = (y_train-ymin)/(ymax-ymin)


# define nn function
def add_layer(inputs,input_size,output_size,activation_function=None):
    with tf.variable_scope("Weights"):
        Weights = tf.Variable(tf.random_normal(shape=[input_size,output_size]),name="weights")
    with tf.variable_scope("biases"):
        biases = tf.Variable(tf.zeros(shape=[1,output_size]) + 0.1,name="biases")
    with tf.name_scope("Wx_plus_b"):
        Wx_plus_b = tf.matmul(inputs,Weights) + biases
    with tf.name_scope("dropout"):
        Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob=keep_prob_s)
    if activation_function is None:
        return Wx_plus_b
    else:
        with tf.name_scope("activation_function"):
            return activation_function(Wx_plus_b)


# set tensorflow part 
xs = tf.placeholder(shape=[None,x_train.shape[1]],dtype=tf.float32,name="inputs")
ys = tf.placeholder(shape=[None,1],dtype=tf.float32,name="y_true")
keep_prob_s = tf.placeholder(dtype=tf.float32, name='keep_prob')

with tf.name_scope("layer_1"):
    l1 = add_layer(xs,54,10,activation_function=tf.nn.relu)
# with tf.name_scope("layer_2"):
#     l2 = add_layer(l1,6,10,activation_function=tf.nn.relu)
with tf.name_scope("y_pred"):
    pred = add_layer(l1,10,1)

# save pred
pred = tf.add(pred,0,name='pred')

# calc loss function and choose training model/param
with tf.name_scope("loss"):
    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred),reduction_indices=[1]))  # mse
    # use l2 regularizer for loss function to prevent overfitting
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred) + tf.contrib.layers.l2_regularizer(.1)(weights),reduction_indices=[1]))
    tf.summary.scalar("loss",tensor=loss)
with tf.name_scope("train"):
    # train_op =tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# draw pics
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(range(50),y_train[0:50],'b')  #展示前50个数据
ax.set_ylim([-2,5])
plt.ion()
plt.show()

# parameters
keep_prob=1  # prevent from overfitting
ITER =5000  # how many times to iter


# define fitting function
def fit(X, y, ax, n, keep_prob):
    init = tf.global_variables_initializer()
    feed_dict_train = {ys: y, xs: X, keep_prob_s: keep_prob}
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="nn_boston_log", graph=sess.graph)  #write tensorbord
        sess.run(init)
        for i in range(n):
            _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)

            if i % 100 == 0:
                print("epoch:%d\tloss:%.5f" % (i, _loss))
                y_pred = sess.run(pred, feed_dict=feed_dict_train)
                rs = sess.run(merged, feed_dict=feed_dict_train)
                writer.add_summary(summary=rs, global_step=i)  #写tensorbord
                saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=i) # save the model
                try:
                    ax.lines.remove(lines[0])
                except:
                    pass
                lines = ax.plot(range(50), y_pred[0:50], 'r--')
                plt.pause(1)

        saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=n)  # save the model

# try to fit the training data
fit(X=x_train,y=y_std,n=ITER,keep_prob=keep_prob,ax=ax)
print('Train Finish.')



def y_scaler(y):
	ymin, ymax = y.min(), y.max() 
	y_std = (y_train-ymin)/(ymax-ymin)
	return y_std

def cross_validation(X,y,keep_prob):
    with tf.Session() as sess:
    	# restore saver
        saver = tf.train.import_meta_graph(meta_graph_or_file="nn_boston_model/nn_boston.model-5000.meta")
        model_file = tf.train.latest_checkpoint(checkpoint_dir="nn_boston_model")
        saver.restore(sess=sess,save_path=model_file)

        # init graph
        graph = tf.get_default_graph()

        # get placeholder from graph
        xs = graph.get_tensor_by_name("inputs:0")
        ys = graph.get_tensor_by_name("y_true:0")
        keep_prob_s = graph.get_tensor_by_name("keep_prob:0")

        # get operation from graph
        pred = graph.get_tensor_by_name("pred:0")

        # run pred
        feed_dict = {xs: X, ys: y, keep_prob_s: keep_prob}
        y_pred = sess.run(pred,feed_dict=feed_dict)

        # calc CV loss
       	print(((y - y_pred)**2).sum()/y.shape[0])
    return

y_cv_std = y_scaler(y=y_cv)
cross_validation(X=x_cv, y=y_cv_std, keep_prob=1)



# define prediction func
def predict(X,keep_prob):
    with tf.Session() as sess:

        # restore saver
        saver = tf.train.import_meta_graph(meta_graph_or_file="nn_boston_model/nn_boston.model-5000.meta")
        model_file = tf.train.latest_checkpoint(checkpoint_dir="nn_boston_model")
        saver.restore(sess=sess,save_path=model_file)

        # init graph
        graph = tf.get_default_graph()

        # get placeholder from graph
        xs = graph.get_tensor_by_name("inputs:0")
        keep_prob_s = graph.get_tensor_by_name("keep_prob:0")

        # get operation from graph
        pred = graph.get_tensor_by_name("pred:0")

        # run pred
        feed_dict = {xs: X, keep_prob_s: keep_prob}
        y_pred = sess.run(pred,feed_dict=feed_dict)
        
    return y_pred.reshape(-1,1)

# try to predict the result
y_pred = predict(X=x_test,keep_prob=1)
