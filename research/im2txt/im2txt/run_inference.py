# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from PIL import Image, ImageOps
from PIL import ImageFont, ImageDraw

import tensorflow as tf
import time
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
import os
#FLAGS = tf.flags.FLAGS

#tf.flags.DEFINE_string("checkpoint_path", "",
#                       "Model checkpoint file or directory containing a "
 #                      "model checkpoint file.")
#tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
#tf.flags.DEFINE_string("input_files", "",
#                       "File pattern or comma-separated list of file patterns "
#                       "of image files.")

#tf.logging.set_verbosity(tf.logging.INFO)
start_time = 0
end_time = 0
check_point_path = "Pretrained-Show-and-Tell-model-master/model.ckpt-2000000"
vocab_file = "Pretrained-Show-and-Tell-model-master/word_counts.txt"
images = os.listdir("Flicker8k_Dataset")
input_files = ",".join(images)
def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               check_point_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(vocab_file)

  filenames = []
  for file_pattern in input_files.split(","):
    filenames.extend(tf.gfile.Glob("Flicker8k_Dataset/"+file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames))
 
  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    start_time = time.time()

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    print("=======================About to iterate==================================")
   # print(filenames)
    for filename in filenames:
      print("=========current:============= ", filename)
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      s = []
      p = []
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        s.append(sentence)
        p.append(math.exp(caption.logprob))
      main_s = s[p.index(max(p))]
      i = 0
      print("  %d) %s (p=%f)" % (i, main_s, max(p)))
      write_caption(main_s, filename,border=(0,0,0,100))
       
    end_time = time.time()
    print("time elapsed {0:.1f} sec".format(end_time - start_time))
def write_caption(caption, image,border):
    
    if isinstance (border, int) or isinstance(border,tuple):
         
         img = Image.open(image)
         img = ImageOps.expand(img, border = border)
         draw = ImageDraw.Draw(img)
         x,y = img.size
         font = ImageFont.truetype('Roboto-Black.ttf',size = 18)
         (x,y) = (5,y-90)
         color = 'rgb(255,255,255)'
         draw.text((x,y), caption,fill=color,font=font)
         img.save('CaptionImage/'+image.split("/")[1])

    
    else:
         raise RuntimeError('Border is not an integer or tuple')



if __name__ == "__main__":
  tf.app.run()
