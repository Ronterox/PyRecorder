from collections import defaultdict
from pynput import mouse, keyboard
import ctypes
import time

ctypes.windll.shcore.SetProcessDpiAwareness(2)

def replay(filename: str, speed: float = 1.0):
    timed_input = defaultdict(lambda: [])

    with open(filename) as f:
        for input_line in f:
            input_line = input_line.split(';')
            second = int(input_line[0])
            ms = float(input_line[1])
            action: str = input_line[2]
            params: list[str] = input_line[3:]
            timed_input[second].append((ms, action, params))

    start = time.time()
    def wait_diff(target: float):
        curr_sec = time.time() - start
        if curr_sec < target / speed:
            time.sleep(target - curr_sec)

    m, k = mouse.Controller(), keyboard.Controller()
    for sec in timed_input:
        wait_diff(sec)
        for ms, action, params in timed_input[sec]:
            wait_diff(ms + sec)
            match action:
                case 'KeyUp' | 'KeyDown':
                    key = keyboard.KeyCode.from_vk(int(params[0]))
                    if action == 'KeyUp':
                        k.release(key)
                    else:
                        k.press(key)
                case 'ButtonUp' | 'ButtonDown':
                    button = mouse.Button[params[0].strip()]
                    if action == 'ButtonUp':
                        m.release(button)
                    else:
                        m.press(button)
                case 'Move':
                    x, _, y = params[0].partition(',')
                    m.position = (int(x), int(y))
                case 'Scroll':
                    dx, _, dy = params[0].partition(',')
                    dx, dy = int(dx), int(dy)

                    if (dx, dy) == (0, 0):
                        dy = 1

                    m.scroll(dx, dy)


if __name__ == '__main__':
    replay('commit.steps')