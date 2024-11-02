import tensorflow as tf

gru_layer = tf.keras.layers.GRU(
    units=64,           
    activation='tanh',  
    recurrent_activation='sigmoid', 
    return_sequences=False,  
    return_state=False,      
    dropout=0.0,             
    recurrent_dropout=0.0    
)

input_tensor = tf.keras.Input(shape=(None, 10)) 

output_tensor = gru_layer(input_tensor)

model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)

model.summary()


