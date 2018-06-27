import tensorflow as tf

from PIL import Image

original_image_list = ['./images/dog1.jpg','./images/dog2.jpg','./images/dog3.jpeg','./images/dog4.jpg']

#make a queue of file names including all the images specified
filename_queue = tf.train.string_input_producer(original_image_list)

#read and entire image file
image_reader = tf.WholeFileReader()

with tf.Session() as sess:
	#coordinate the loading of image files
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	image_list = []
	for i in range(len(original_image_list)):
		#read a whole file from th queue, the first returned value in the tuple is the filename which we are ignoring
		_, image_file = image_reader.read(filename_queue)

		#decode the image as a JPEG file, this will turn it into a tensor which we can then use in training
		image = tf.image.decode_jpeg(image_file)

		#get a tensor of resized images
		image = tf.image.resize_images(image, [224,224])
		image.set_shape((224,224,3))

		image = tf.image.flip_up_down(image)

		image = tf.image.central_crop(image,central_fraction=0.5)

		#get an image tensor and print its value
		image_array = sess.run(image)
		print image_array.shape

		#converts a numpy array of the kind (224,224,3) to a tensor of shape (224,224,30
		image_tensor = tf.stack(image_array)

		print image_tensor
		image_list.append(image_tensor)

	#finish off the filename queue coordinator
	coord.request_stop()
	coord.join(threads)

	#convers all tensors to a single tensor with a 4th dimension
	#4 images of (224,224,3) can be accessed as (0,224,224,3), (1,224,224,3) ...
	images_tensor = tf.stack(image_list)
	print images_tensor

	summary_writer = tf.summary.FileWriter('.',graph=sess.graph)

	#write out all the images in one go
	summary_str = sess.run(tf.summary.image('images', images_tensor))
	summary_writer.add_summary(summary_str)

	summary_writer.close()
