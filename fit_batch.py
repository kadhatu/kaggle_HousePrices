def fit(X, y, n, keep_prob):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logdir="nn_boston_log", graph=sess.graph)  #write tensorbord
        sess.run(init)
        for i in range(n):
            # set param for training with batch
            batch_size = 500
            dataset_size = X.shape[0]
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            
            feed_dict_train = {ys: y[start:end], xs: X[start:end], keep_prob_s:keep_prob}
            sess.run(train_op, feed_dict=feed_dict_train)
            
            if i % 1000 == 0:
              _loss = sess.run(loss, feed_dict={ys: y, xs: X, keep_prob_s:keep_prob})
            	print("epoch:%d\tloss:%.5f" % (i, _loss))
            	y_pred = sess.run(pred, feed_dict=feed_dict_train)
            	rs = sess.run(merged, feed_dict=feed_dict_train)
            	writer.add_summary(summary=rs, global_step=i)  #write tensorbord
            	saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=i) # save model
              
        saver.save(sess=sess, save_path="nn_boston_model/nn_boston.model", global_step=n)  # save model
        
# fit the training data without plot figure
fit(X=x_train,y=y_std,n=ITER,keep_prob=keep_prob)
