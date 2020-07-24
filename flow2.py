#///-----------------------------------------------------------\\\
#//                          Setup                              \\

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import model
import collections
import common
import gstreamer
import numpy as np
import operator
import os
import re
import svgwrite
import time
from periphery import GPIO

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, GstBase, Gtk

Category = collections.namedtuple('Category', ['id', 'score'])

gpio6 = GPIO(6, "in")
gpio7 = GPIO(7, "out")
gpio8 = GPIO(8, "out")

access = 0
answer = 0
house = False
parcel = False

#\\                                                             //
#\\\-----------------------------------------------------------///


#///-----------------------------------------------------------\\\
#//                      Listening                              \\

def print_results(result, commands, labels, top=3):
  """Example callback function that prints the passed detections."""
  global answer
  answered = False
  top_results = np.argsort(-result)[:top]
  for p in range(top):
    l = labels[top_results[p]]
    if gpio6.read() == True:
      answered = True
      answer = 3
    elif l == "yes" and result[top_results[p]] > 0.2:
      answered = True
      answer = 1
    elif l == "no" and result[top_results[p]] > 0.01:
      answered = True
      answer = 2
    if l in commands.keys():
      threshold = commands[labels[top_results[p]]]["conf"]
    else:
      threshold = 0.5
    if top_results[p] and result[top_results[p]] > threshold:
      sys.stdout.write("\033[1m\033[93m*%15s*\033[0m (%.3f)" %
                       (l, result[top_results[p]]))
    elif result[top_results[p]] > 0.005:
      sys.stdout.write(" %15s (%.3f)" % (l, result[top_results[p]]))
  sys.stdout.write("\n")
  return answered

#\\                                                             //
#\\\-----------------------------------------------------------///


def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def generate_svg(size, text_lines):
    dwg = svgwrite.Drawing('', size=size)
    for y, line in enumerate(text_lines, start=1):
      dwg.add(dwg.text(line, insert=(11, y*20+1), fill='black', font_size='20'))
      dwg.add(dwg.text(line, insert=(10, y*20), fill='white', font_size='20'))
    return dwg.tostring()

def get_output(interpreter, top_k, score_threshold):
    """Returns no more than top_k categories with score >= score_threshold."""
    scores = common.output_tensor(interpreter, 0)
    categories = [
        Category(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(categories, key=operator.itemgetter(1), reverse=True)

def main():


#///-----------------------------------------------------------\\\
#//                    Scanning Image                           \\

    def user_callback(input_tensor, src_size, inference_box):
        global access
        global house
        global parcel
        nonlocal fps_counter
        start_time = time.monotonic()
        common.set_input(interpreter, input_tensor)
        interpreter.invoke()
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        results = get_output(interpreter, args.top_k, args.threshold)
        end_time = time.monotonic()
        text_lines = [
            ' ',
            'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
            'FPS: {} fps'.format(round(next(fps_counter))),
        ]
        for result in results:
            text_lines.append('score={:.2f}: {}'.format(result.score, labels.get(result.id, result.id)))
            if gpio6.read() == True:
                access = 2
                Gtk.main_quit()    
            elif house:
                if labels.get(result.id, result.id) == "tree frog, tree-frog" and result.score > 0.3:
                    access = 1
                    Gtk.main_quit()
                elif (labels.get(result.id, result.id) == "acoustic guitar" or labels.get(result.id, result.id) == "jigsaw puzzle" or labels.get(result.id, result.id) == "jellyfish" or labels.get(result.id, result.id) == "basketball" or labels.get(result.id, result.id) == "soccer ball") and result.score > 0.3:
                    access = 0
                    Gtk.main_quit()
            elif parcel:
                if labels.get(result.id, result.id) == "acoustic guitar" and result.score > 0.3: 
                    access = 1
                    Gtk.main_quit()
                elif (labels.get(result.id, result.id) == "tree frog, tree-frog" or labels.get(result.id, result.id) == "jigsaw puzzle" or labels.get(result.id, result.id) == "jellyfish" or labels.get(result.id, result.id) == "basketball" or labels.get(result.id, result.id) == "soccer ball") and result.score > 0.3:
                    access = 0
                    Gtk.main_quit()
                
        print(' '.join(text_lines))
        return generate_svg(src_size, text_lines)
      
#\\                                                             //
#\\\-----------------------------------------------------------///

    while(1):
        global access
        global answer
        global house
        global parcel
        
        gpio7.write(True)
        gpio8.write(True)
        while(gpio6.read() == False):                                     #  Waiting for signal
          time.sleep(0.05)
        time.sleep(2)
        
        # Setting up voice recogniton
        parser = argparse.ArgumentParser()
        model.add_model_flags(parser)
        args = parser.parse_args()
        interpreter = model.make_interpreter(args.model_file)
        interpreter.allocate_tensors()
        mic = args.mic if args.mic is None else int(args.mic)
        model.classify_audio(mic, interpreter, 1,                           # Calling Listening Function
                     labels_file="config/labels_gc2.raw.txt",
                     result_callback=print_results,
                     sample_rate_hz=int(args.sample_rate_hz),
                     num_frames_hop=int(args.num_frames_hop))
        
        if answer == 3:     # Timed out
            answer = 0
            house = False
            parcel = False
        elif answer == 1:   # Yes
            gpio8.write(True)
            gpio7.write(False)
            while(gpio6.read() == False):
              time.sleep(0.05)
            gpio7.write(True)
            answer = 0
            house = True
            parcel = False
            
        elif answer == 2:   # No
            gpio8.write(False)
            gpio7.write(False)
            while(gpio6.read() == False):
              time.sleep(0.05)
            gpio7.write(True)
            answer = 0
            house = False
            time.sleep(1)
            model.classify_audio(mic, interpreter, 2,                           # Calling Listening Function
                        labels_file="config/labels_gc2.raw.txt",
                        result_callback=print_results,
                        sample_rate_hz=int(args.sample_rate_hz),
                        num_frames_hop=int(args.num_frames_hop))
            if answer == 3:     # Timed out
                answer = 0
                parcel = False
            elif answer == 1:   # Yes
                gpio8.write(True)
                gpio7.write(False)
                while(gpio6.read() == False):
                  time.sleep(0.05)
                gpio7.write(True)
                answer = 0
                parcel = True
            elif answer == 2:   # No
                gpio8.write(False)
                gpio7.write(False)
                while(gpio6.read() == False):
                  time.sleep(0.05)
                gpio7.write(True)
                answer = 0
                parcel = False
        if house or parcel:
            # Setting up image recogniton
            default_model_dir = '../all_models'
            default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
            default_labels = 'imagenet_labels.txt'
            parser = argparse.ArgumentParser()
            parser.add_argument('--model', help='.tflite model path',
                                default=os.path.join(default_model_dir,default_model))
            parser.add_argument('--labels', help='label file path',
                                default=os.path.join(default_model_dir, default_labels))
            parser.add_argument('--top_k', type=int, default=3,
                                help='number of categories with highest score to display')
            parser.add_argument('--threshold', type=float, default=0.1,
                                help='classifier score threshold')
            parser.add_argument('--videosrc', help='Which video source to use. ',
                                default='/dev/video0')
            parser.add_argument('--videofmt', help='Input video format.',
                                default='raw',
                                choices=['raw', 'h264', 'jpeg'])
            args = parser.parse_args()

            print('Loading {} with {} labels.'.format(args.model, args.labels))
            interpreter = common.make_interpreter(args.model)
            interpreter.allocate_tensors()
            labels = load_labels(args.labels)

            w, h, _  = common.input_image_size(interpreter)
            inference_size = (w, h)
            # Average fps over last 30 frames.
            fps_counter = common.avg_fps_counter(30)
            result = gstreamer.run_pipeline(user_callback,                # Calling Scanning Image Function
                                        src_size=(640, 480),
                                        appsink_size=inference_size,
                                        videosrc=args.videosrc,
                                        videofmt=args.videofmt)
                
            # Communication with ESP32 Board
            if access == 1:
                gpio8.write(True)
                gpio7.write(False)
                while(gpio6.read() == False):
                    time.sleep(0.05)
                gpio7.write(True)
            elif access == 0:
                gpio8.write(False)
                gpio7.write(False)
                while(gpio6.read() == False):
                    time.sleep(0.05)
                gpio7.write(True)
        
        time.sleep(2)


if __name__ == '__main__':
    main()
