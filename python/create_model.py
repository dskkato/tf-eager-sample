import tensorflow as tf

# default input shape 224x224x3
model = tf.keras.applications.MobileNetV3Small(
    input_shape=[224, 224, 3], weights="imagenet"
)

# save the model
directory = "model"
model.save(directory, save_format="tf")

######################################################
# Check the prediction results for the sample image. #
######################################################
# load sample image
fname = "sample_images/macaque.jpg"
buf = tf.io.read_file(fname)
img = tf.image.decode_jpeg(buf)

small = tf.image.resize(img, (224, 224), method="lanczos3", antialias=True)
# check model prediction
predict = model(small[tf.newaxis, :, :, :])
predict = predict.numpy()
decoded = tf.keras.applications.imagenet_utils.decode_predictions(predict, top=1)[0]

print(f"""
argmax={predict.argmax(axis=1)[0]}
""")
print("class_name | class_description | score")
print("-----------+-------------------+------")
print(f"{decoded[0][0]:>10} | {decoded[0][1]:>17} | {decoded[0][2]:0.3f}")