#!/usr/bin/python
#-*- encoding: utf8 -*-

from __future__ import division

import contextlib
import functools
import re
import signal
import sys
import threading
from array import array


import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
from google.cloud.proto.speech.v1beta1 import cloud_speech_pb2
from google.rpc import code_pb2
import grpc
import pyaudio
from six.moves import queue
from ctypes import *

import rospy
from std_msgs.msg import String, Float64, Empty
from mhri_social_msgs.msg import RecognizedWord


RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
DEADLINE_SECS = 60 * 5 + 5
SPEECH_SCOPE = 'https://www.googleapis.com/auth/cloud-platform'


ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
	pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

@contextlib.contextmanager
def noalsaerr():
	asound = cdll.LoadLibrary('libasound.so')
	asound.snd_lib_error_set_handler(c_error_handler)
	yield
	asound.snd_lib_error_set_handler(None)


class GoogleCloudSpeechNode:
    def __init__(self):
        rospy.init_node('google_speech_node', anonymous=False)

        self.is_speaking_started = False
        self.published_started = False
        self.count_silency_time = 0

        self.pub_recognized_word = rospy.Publisher('recognized_word', RecognizedWord, queue_size=10)
        self.pub_start_speech = rospy.Publisher('start_of_speech', Empty, queue_size=10)
        self.pub_end_speech = rospy.Publisher('end_of_speech', Empty, queue_size=10)
        self.pub_silency_detected = rospy.Publisher('silency_detected', Empty, queue_size=10)

        self.service = cloud_speech_pb2.SpeechStub(self.make_channel('speech.googleapis.com', 443))
        self.t = threading.Thread(target=self.thread_streaming)
        self.t.start()

        rospy.spin()

    def thread_streaming(self):
        while not rospy.is_shutdown():
            with self.record_audio(RATE, CHUNK) as buffered_audio_data:
                requests = self.request_stream(buffered_audio_data, RATE)
                self.recognize_stream = self.service.StreamingRecognize(requests, DEADLINE_SECS)

                try:
                    self.listen_print_loop(self.recognize_stream)
                    self.recognize_stream.cancel()
                except grpc.RpcError as e:
                    code = e.code()
                    if code is not code.CANCELLED:
                        raise

    def make_channel(self, host, port):
        credentials, _ = google.auth.default(scopes=[SPEECH_SCOPE])
        http_request = google.auth.transport.requests.Request()
        target = '{}:{}'.format(host, port)

        return google.auth.transport.grpc.secure_authorized_channel(credentials, http_request, target)

    def request_stream(self, data_stream, rate, interim_results=True):
        recognition_config = cloud_speech_pb2.RecognitionConfig(
            encoding='LINEAR16',  # raw 16-bit signed LE samples
            sample_rate=rate,  # the rate in hertz
            language_code='ko-KR',  # a BCP-47 language tag
        )
        streaming_config = cloud_speech_pb2.StreamingRecognitionConfig(
            interim_results=interim_results,
            single_utterance=False,
            config=recognition_config,
        )

        yield cloud_speech_pb2.StreamingRecognizeRequest(streaming_config=streaming_config)
        for data in data_stream:
            yield cloud_speech_pb2.StreamingRecognizeRequest(audio_content=data)

    @contextlib.contextmanager
    def record_audio(self, rate, chunk):
        buff = queue.Queue()

        with noalsaerr():
            audio_interface = pyaudio.PyAudio()
        audio_stream = audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
            stream_callback=functools.partial(self._fill_buffer, buff),
        )
        yield self._audio_data_generator(buff)

        audio_stream.stop_stream()
        audio_stream.close()
        buff.put(None)
        audio_interface.terminate()

    def _fill_buffer(self, buff, in_data, frame_count, time_info, status_flags):
        data_chunk = array('h', in_data)
        if max(data_chunk) < 1000:
            self.count_silency_time += 1
            if self.count_silency_time > 50:
                self.pub_silency_detected.publish()
                self.count_silency_time = 0
                self.recognize_stream.cancel()
                rospy.logdebug("Silency detected...")
        else:
            self.count_silency_time = 0

        buff.put(in_data)
        return None, pyaudio.paContinue

    def _audio_data_generator(self, buff):
        stop = False
        while not stop:
            data = [buff.get()]
            while True:
                try:
                    data.append(buff.get(block=False))
                except queue.Empty:
                    break

            if None in data:
                stop = True
                data.remove(None)

            yield b''.join(data)

    def listen_print_loop(self, recognize_stream):
        num_chars_printed = 0
        for resp in recognize_stream:
            if resp.endpointer_type == 2 and self.is_speaking_started == False:
                if not resp.results:
                    self.recognize_stream.cancel()

            if resp.error.code != code_pb2.OK:
                raise RuntimeError('Server error: ' + resp.error.message)

            if not resp.results:
                continue

            self.is_speaking_started = True
            if not self.published_started:
                self.pub_start_speech.publish()
                rospy.logdebug("Speech started...")
                self.published_started = True

            result = resp.results[0]
            transcript = result.alternatives[0].transcript
            confidence = result.alternatives[0].confidence            

            if not result.is_final:
                rospy.logdebug("I'm listening...")
            else:
                self.is_speaking_started = False
                self.pub_end_speech.publish()
                rospy.logdebug("Speech stoped...")
                self.published_started = False

                result = RecognizedWord()
                result.recognized_word = transcript
                result.confidence = confidence

                self.pub_recognized_word.publish(result)
                num_chars_printed = 0
                break


if __name__ == '__main__':
    m = GoogleCloudSpeechNode()
