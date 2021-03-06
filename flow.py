# To add: higher rating requiment for pictures to minimize false triggers
#         loading recording software before speaking to speed up process
#         test with good setup

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
import simpleaudio as sa
import VL53L0X

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('Gtk', '3.0')
from gi.repository import GLib, GObject, Gst, GstBase, Gtk

Category = collections.namedtuple('Category', ['id', 'score'])

gpio6 = GPIO(6, "out")
gpio73 = GPIO(73, "out")
gpio7 = GPIO(7, "out")
gpio8 = GPIO(8, "out")

motion = VL53L0X.VL53L0X()

access = 0
answer = 0
house = False
parcel = False

def print_results(result, commands, labels, top=3):
  """Example callback function that prints the passed detections."""
  global answer
  answered = False
  top_results = np.argsort(-result)[:top]
  for p in range(top):
    l = labels[top_results[p]]
    if l == "yes":
      answered = True
      answer = 1
    elif l == "no":
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
            if house:
                if labels.get(result.id, result.id) == "tree frog, tree-frog":
                    access = 1
                    gpio6.write(True)
                    Gtk.main_quit()
                elif labels.get(result.id, result.id) == "acoustic guitar" or labels.get(result.id, result.id) == "jigsaw puzzle" or labels.get(result.id, result.id) == "jellyfish" or labels.get(result.id, result.id) == "basketball" or labels.get(result.id, result.id) == "soccer ball":
                    access = 0
                    gpio73.write(True)
                    Gtk.main_quit()
            elif parcel:
                if labels.get(result.id, result.id) == "acoustic guitar": 
                    access = 1
                    gpio7.write(True)
                    Gtk.main_quit()
                elif labels.get(result.id, result.id) == "tree frog, tree-frog" or labels.get(result.id, result.id) == "jigsaw puzzle" or labels.get(result.id, result.id) == "jellyfish" or labels.get(result.id, result.id) == "basketball" or labels.get(result.id, result.id) == "soccer ball":
                    access = 0
                    gpio8.write(True)
                    Gtk.main_quit()
                
        print(' '.join(text_lines))
        return generate_svg(src_size, text_lines)

    while(1):
        global access
        global answer
        global house
        global parcel
        
        gpio6.write(False)
        gpio73.write(False)
        gpio7.write(False)
        gpio8.write(False)

        motion.start_ranging(VL53L0X.VL53L0X_BETTER_ACCURACY_MODE)
        timing = motion.get_timing()
        if (timing < 20000):
            timing = 20000
        distance = motion.get_distance()
        while(distance > 500):
            distance = motion.get_distance()
            time.sleep(timing/1000000.00)
        motion.stop_ranging()

        wave_obj = sa.WaveObject.from_wave_file("welcome.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        wave_obj = sa.WaveObject.from_wave_file("entry.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        
        # Voice Recognition
        parser = argparse.ArgumentParser()
        model.add_model_flags(parser)
        args = parser.parse_args()
        interpreter = model.make_interpreter(args.model_file)
        interpreter.allocate_tensors()
        mic = args.mic if args.mic is None else int(args.mic)
        model.classify_audio(mic, interpreter, 1,
                     labels_file="config/labels_gc2.raw.txt",
                     result_callback=print_results,
                     sample_rate_hz=int(args.sample_rate_hz),
                     num_frames_hop=int(args.num_frames_hop))
        if answer == 1:
            wave_obj = sa.WaveObject.from_wave_file("key.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()
            answer = 0
            house = True
            parcel = False
        elif answer == 2:
            wave_obj = sa.WaveObject.from_wave_file("package.wav")
            play_obj = wave_obj.play()
            play_obj.wait_done()
            answer = 0
            house = False
            # Voice Recognition
            model.classify_audio(mic, interpreter, 2,
                        labels_file="config/labels_gc2.raw.txt",
                        result_callback=print_results,
                        sample_rate_hz=int(args.sample_rate_hz),
                        num_frames_hop=int(args.num_frames_hop))
            if answer == 1:
                wave_obj = sa.WaveObject.from_wave_file("key.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()
                answer = 0
                parcel = True
            elif answer == 2:
                wave_obj = sa.WaveObject.from_wave_file("goodday.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()
                answer = 0
                parcel = False
        if house or parcel:
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
            result = gstreamer.run_pipeline(user_callback,
                                        src_size=(640, 480),
                                        appsink_size=inference_size,
                                        videosrc=args.videosrc,
                                        videofmt=args.videofmt)
            if access:
                if house:
                    wave_obj = sa.WaveObject.from_wave_file("stay.wav")
                elif parcel:
                    wave_obj = sa.WaveObject.from_wave_file("parcel.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()
            else:
                wave_obj = sa.WaveObject.from_wave_file("denied.wav")
                play_obj = wave_obj.play()
                play_obj.wait_done()
        
        time.sleep(3)


if __name__ == '__main__':
    main()
    
