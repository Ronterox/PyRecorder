from pynput import mouse, keyboard
from replay import replay
from math import floor

import logging
import ctypes
import time

ctypes.windll.shcore.SetProcessDpiAwareness(2)

filename = "mouse_log.txt"
logging.basicConfig(filename=filename, filemode='w', level=logging.DEBUG, format='%(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

print('Recording...')
recording = True
moves = 0
start = time.time()

def timestamp() -> tuple[int, float]:
    curr_time = time.time() - start
    floor_time = floor(curr_time)
    return floor_time, curr_time - floor_time

def on_move(x: int, y: int):
    global recording, moves
    moves += 1
    if not recording or moves % 5 != 0:
        return
    t, ms = timestamp()
    logging.info(f'{t};{ms};Move;{int(x)},{int(y)}')

def on_click(x: int, y: int, button: mouse.Button, pressed: bool):
    global recording
    if not recording:
        return

    click = 'ButtonDown' if pressed else 'ButtonUp'
    t, ms = timestamp()
    logging.info(f'{t};{ms};{click};{int(x)},{int(y)};{button.name}')

Key = keyboard.Key | keyboard.KeyCode
def on_press(key: Key):
    global recording
    if keyboard.Key.f10 == key:
        recording = False
        print('Finished recording...')
        return

    vk = key.value.vk if isinstance(key, keyboard.Key) else key.vk
    t, ms = timestamp()
    logging.info(f'{t};{ms};KeyDown;{vk};{key}')

def on_release(key: Key):
    global recording
    if not recording:
        return

    vk = key.value.vk if isinstance(key, keyboard.Key) else key.vk
    t, ms = timestamp()
    logging.info(f'{t};{ms};KeyUp;{vk};{key}')

mouse_thread = mouse.Listener(on_click=on_click, on_move=on_move)
keyboard_thread = keyboard.Listener(on_press=on_press, on_release=on_release)

mouse_thread.start()
keyboard_thread.start()

while recording:
    time.sleep(0.5)

mouse_thread.stop()
keyboard_thread.stop()

replay(filename)
