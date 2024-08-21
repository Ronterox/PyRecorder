from pynput import mouse, keyboard
from replay import replay
from math import floor
from mss import mss

import multiprocessing as mp
import mss.tools
import logging
import time
import os

import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(2)
user32 = ctypes.windll.user32
width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

directory = 'screenshots'
os.makedirs(directory, exist_ok=True)
filename = "osu.steps"

logging.basicConfig(filename=filename, filemode='w', level=logging.DEBUG, format='%(message)s')
logging.getLogger().addHandler(logging.StreamHandler())

print('Recording...')
recording = True
moves = 0
start = time.time()

def screenshot(name: str):
    global mp_queue

    monitor = {'top': 0, 'left': 0, 'width': width, 'height': height}
    with mss.mss() as sct:
        sct_img = sct.grab(monitor)
        mp_queue.put((sct_img.rgb, sct_img.size, f'{directory}/{name}.png'))

def save_screenshot(queue: mp.Queue):
    data = queue.get()
    while data is not None:
        (rgb, size, name) = data
        mss.tools.to_png(rgb, size, level=9, output=name)
        data = queue.get()

def timestamp() -> tuple[int, float]:
    curr_time = time.time() - start
    floor_time = floor(curr_time)
    return floor_time, curr_time - floor_time


def on_scroll(_x: int, _y: int, dx: int, dy: int):
    global recording
    if not recording:
        return
    t, ms = timestamp()
    logging.info(f'{t};{ms};Scroll;{dx},{dy}')


def on_move(x: int, y: int):
    global recording, moves
    moves += 1
    if not recording or moves % 5 != 0:
        return
    t, ms = timestamp()
    logging.info(f'{t};{ms};Move;{int(x)},{int(y)}')
    if moves % 15 == 0:
        screenshot(t)


def on_click(_x: int, _y: int, button: mouse.Button, pressed: bool):
    global recording
    if not recording:
        return

    click = 'ButtonDown' if pressed else 'ButtonUp'
    t, ms = timestamp()
    logging.info(f'{t};{ms};{click};{button.name}')
    screenshot(t)


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


if __name__ == '__main__':
    mp_queue = mp.Queue()

    mouse_thread = mouse.Listener(on_click=on_click, on_move=on_move, on_scroll=on_scroll)
    keyboard_thread = keyboard.Listener(on_press=on_press, on_release=on_release)

    mouse_thread.start()
    keyboard_thread.start()

    proc = mp.Process(target=save_screenshot, args=(mp_queue,))
    proc.start()

    while recording:
        time.sleep(0.5)

    mouse_thread.stop()
    keyboard_thread.stop()

    mp_queue.put(None)
    proc.join()

    print("Ctrl+C to Cancel, Replaying in", end=' ')
    for i in range(5, 0, -1):
        print(f"{i}", end=', ')
        time.sleep(1)

    replay(filename)
