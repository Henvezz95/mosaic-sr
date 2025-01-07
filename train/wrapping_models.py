import tensorflow as tf


class simple_model(tf.keras.Model):
    def __init__(self, cnn, use_mask=False, var_pred = True, **kwargs):
        super(simple_model, self).__init__(**kwargs)
        self.cnn = cnn
        self.use_mask = use_mask
        self.var_pred = var_pred
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mse_tracker = tf.keras.metrics.Mean(name="wmse")
        self.val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")
        self.val_mse_tracker = tf.keras.metrics.Mean(name="val_wmse")

    @property
    def metrics(self):
        return [
        self.loss_tracker,
        self.mse_tracker,
        self.val_loss_tracker          
        ]
    
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x,y_true,weights,_ = data
            
            y_pred = self.cnn(x, training=True)
            #Loss with variance prediction
            mse = tf.keras.losses.MSE(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])
            if self.use_mask:
                #The total loss function is the weighted mean of the pixel-wise cost function
                cost_function = tf.math.multiply(cost_function, weights[:,:,:,0])
                weighted_mse = tf.math.multiply(mse, weights[:,:,:,0])*10
            else:
                weighted_mse = mse*10
            if self.var_pred:
                cost_function = tf.math.divide(mse,y_pred[:,:,:,1])
                log_var = tf.math.log(y_pred[:,:,:,1]+1e-12)
                cost_function += log_var
                distribution_loss = tf.reduce_mean(cost_function)
                loss = weighted_mse + distribution_loss/1e5
            else:
                loss = weighted_mse
        
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.mse_tracker.update_state(weighted_mse)
        return {
            "loss": self.loss_tracker.result(),
            "wmse":self.mse_tracker.result()
        }
    
    def test_step(self, data):
        x,y_true,weights,_ = data
        y_pred = self.cnn(x, training=True)
        #Loss with variance prediction
        mse = tf.keras.losses.MSE(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])
        log_var = tf.math.log(y_pred[:,:,:,1]+1e-12)
        cost_function = tf.math.divide(mse,y_pred[:,:,:,1]) + log_var
        distribution_loss = tf.reduce_mean(cost_function)
        
        if self.use_mask:
            #The total loss function is the weighted mean of the pixel-wise cost function
            cost_function = tf.math.multiply(cost_function, weights[:,:,:,0])
            weighted_mse = tf.math.multiply(mse, weights[:,:,:,0])*10
        else:
            weighted_mse = mse*10
        loss = weighted_mse + distribution_loss/1e5
        
        self.val_loss_tracker.update_state(loss)
        self.val_mse_tracker.update_state(weighted_mse)

        return {
            "val_loss": self.val_loss_tracker.result(),
            "val_wmse": self.val_mse_tracker.result()
        }
    
class iterative_model(tf.keras.Model):
    def __init__(self, aggregate_model, use_mask=True, **kwargs):
        super(iterative_model, self).__init__(**kwargs)
        self.aggregate_model = aggregate_model
        self.loss_weights = [1,1,1,1]
        self.weight_pointer = 0
        self.use_mask = use_mask
        self.loss_trackers = [tf.keras.metrics.Mean(name="total_loss")]
        self.wmse_trackers = []
        self.val_loss_trackers = [tf.keras.metrics.Mean(name="total_val_loss")]
        
        
        i = 1
        for out in aggregate_model.output:
            self.loss_trackers.append(tf.keras.metrics.Mean(name="loss_"+str(i)))
            self.wmse_trackers.append(tf.keras.metrics.Mean(name="wmse_"+str(i)))
            self.val_loss_trackers.append(tf.keras.metrics.Mean(name="val_loss_"+str(i)))
            i+=1

    @property
    def metrics(self):
        return [
        *self.loss_trackers,
        *self.wmse_trackers,
        *self.val_loss_trackers
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            x,y_true,weights,_ = data
            total_loss = 0
            predictions = self.aggregate_model(x, training=True)
            i=0
            for y_pred in predictions:
                if y_pred.shape[-1] == 2:
                    #Loss with variance prediction
                    mse = tf.keras.losses.MSE(y_true[:,:,:,0:1], y_pred[:,:,:,0:1])
                    log_var = tf.math.log(y_pred[:,:,:,1]+1e-12)
                    cost_function = tf.math.divide(mse,y_pred[:,:,:,1]) + log_var
                    
                    if self.use_mask:
                        #The total loss function is the weighted mean of the pixel-wise cost function
                        cost_function = tf.math.multiply(cost_function, weights[:,:,:,0])
                        distribution_loss = tf.reduce_mean(cost_function)
                        weighted_mse = tf.math.multiply(mse, weights[:,:,:,0])*10
                    else:
                        distribution_loss = tf.reduce_mean(cost_function)
                        weighted_mse = mse*10

                    self.wmse_trackers[i].update_state(weighted_mse)
                    loss = (distribution_loss/1e5+weighted_mse)
                    self.loss_trackers[i+1].update_state(loss)
                    total_loss += loss*self.loss_weights[i]
                else:
                    #Loss without variance prediction
                    mse = tf.keras.losses.MSE(y_true, y_pred)
                    if self.use_mask:
                        custom_values = tf.math.multiply(mse, weights[:,:,:,0])
                        loss = tf.reduce_mean(custom_values)*10
                    else:
                        loss = tf.reduce_mean(mse)*10
                    self.loss_trackers[i+1].update_state(loss)
                    self.wmse_trackers[i].update_state(loss)
                    total_loss += loss*self.loss_weights[i]
                i+=1
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_trackers[0].update_state(total_loss)
        
        keys = ['total_loss']
        values = [self.loss_trackers[0].result()]
        for i in range(0, len(predictions)):
            keys.append('loss_'+str(i+1))
            values.append(self.loss_trackers[i+1].result())
        for i in range(0, len(predictions)):
            keys.append('wmse_'+str(i+1))
            values.append(self.wmse_trackers[i].result())
        loss_dictionary = dict(zip(keys, values))

        
        return loss_dictionary
    def test_step(self, data):
        x,y_true,weights,_ = data
        total_loss = 0
        predictions = self.aggregate_model(x, training=False)
        i=1
        for y_pred in predictions:
            mse = tf.keras.losses.MSE(y_true, y_pred)
            custom_values = tf.math.multiply(mse, weights[:,:,:,0])
            loss = tf.reduce_mean(custom_values)*10
            self.val_loss_trackers[i].update_state(loss)
            total_loss += loss
            i+=1
        
        self.val_loss_trackers[0].update_state(total_loss)

        return {
            "loss": self.val_loss_trackers[0].result()
        }