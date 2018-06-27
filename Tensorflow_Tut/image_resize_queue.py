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

		#get an image tensor and print its value
		image_array = sess.run(image)
		print image_array.shape

		Image.fromarray(image_array.astype('uint8'), 'RGB').show()

		#the expand_dims adds a new dimension
		image_list.append(tf.expand_dims(image_array, 0))

	#finish off the filename queue coordinator
	coord.request_stop()
	coord.join(threads)
	
	index = 0

	#write image summary
	summary_writer = tf.summary.FileWriter('.',graph=sess.graph)

	for image_tensor in image_list:
		summary_str = sess.run(tf.summary.image('image-' + str(index), image_tensor))
		summary_writer.add_summary(summary_str)
		index += 1
	summary_writer.close()

