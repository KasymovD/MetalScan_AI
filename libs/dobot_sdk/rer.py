# -*- coding: utf-8 -*-
import time, os
from pathlib import Path
import DobotDllType as dType

# --- DLL рядом со скриптом ---
HERE = Path(__file__).resolve().parent
os.add_dll_directory(str(HERE))

# -------- ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ --------
PORT, BAUD = "", 115200          # "" = авто; иначе "COM3"

# -------- ПАРАМЕТРЫ ДВИЖЕНИЯ --------
SPEED_TRANSIT  = (80, 80)        # быстрые перегоны
SPEED_APPROACH = (40, 40)        # подлет
SPEED_FINE     = (20, 20)        # посадка / отрыв

Z_LIFT          = 50.0           # подлёт сверху
APPROACH_BUF    = 20.0           # буфер над рабочей Z
SETTLE_DWELL    = 0.25           # пауза перед/после grip, с

Z_PICK_OFFS     = 0.0            # тонкая доводка Z в A
Z_PLACE_OFFS    = 0.0            # тонкая доводка Z в B

# -------- КООРДИНАТЫ TCP (твои замеры) --------
A1 = (198.5629,  -32.3904,  -20.1180,  -9.2647)
A2 = (191.9371,   22.5373,  -20.5744,  -9.2647)
A3 = (186.7198,   79.5747,  -20.5744,  -9.2647)

B  = ( 52.4785, -268.3073,   22.8921, -100.3589)
SAFE = (-134.9857, -270.4609, 52.2684, -116.5236)

# ----------------------------------------------

def start_queue(api):
    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api); time.sleep(0.05)

def stop_queue(api):
    dType.SetQueuedCmdStopExec(api);  time.sleep(0.05)

def wait_done(api, idx, tout=30):
    t0 = time.time()
    while dType.GetQueuedCmdCurrentIndex(api)[0] < idx:
        if time.time() - t0 > tout:
            raise TimeoutError("Queue timeout")
        time.sleep(0.02)

def set_speed(api, vel, acc):
    dType.SetPTPCommonParams(api, vel, acc, isQueued=0); time.sleep(0.02)

def ptp(api, x, y, z, r):
    idx = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, x, y, z, r, 1)[0]
    wait_done(api, idx)

def descend(api, x, y, z, r, offset=0.0):
    # быстро над точку
    set_speed(api, *SPEED_TRANSIT);  ptp(api, x, y, z + Z_LIFT, r)
    # медленно в буфер
    set_speed(api, *SPEED_APPROACH); ptp(api, x, y, z + max(APPROACH_BUF, 5), r)
    # очень медленно на посадку
    set_speed(api, *SPEED_FINE);     ptp(api, x, y, z + offset, r)

def ascend(api, x, y, z, r):
    # мягкий отрыв
    set_speed(api, *SPEED_FINE);     ptp(api, x, y, z + max(APPROACH_BUF, 5), r)
    # быстрый подъём
    set_speed(api, *SPEED_TRANSIT);  ptp(api, x, y, z + Z_LIFT, r)

def grip(api, on: bool):
    idx = dType.SetEndEffectorGripper(api, 1, 1 if on else 0, 1)[0]
    wait_done(api, idx)

def home(api):
    stop_queue(api); dType.ClearAllAlarmsState(api); start_queue(api)
    idx = dType.SetHOMECmd(api, 0, 1)[0]; wait_done(api, idx, 45)

def connect():
    api = dType.load()
    if dType.ConnectDobot(api, PORT, BAUD)[0]:
        raise RuntimeError("Не удаётся подключиться (закрой DobotStudio)")
    dType.SetCmdTimeout(api, 3000); start_queue(api)
    set_speed(api, *SPEED_TRANSIT)
    return api

def cycle_slot(api, Ax, By):
    ax, ay, az, ar = Ax
    bx, by, bz, br = By

    # A -> B
    descend(api, ax, ay, az, ar, Z_PICK_OFFS)
    time.sleep(SETTLE_DWELL); grip(api, True); time.sleep(SETTLE_DWELL)
    ascend(api, ax, ay, az, ar)

    descend(api, bx, by, bz, br, Z_PLACE_OFFS)
    time.sleep(SETTLE_DWELL); grip(api, False); time.sleep(SETTLE_DWELL)
    ascend(api, bx, by, bz, br)

    ptp(api, *SAFE)

    input("[ENTER] — забрать из B и вернуть…")

    # B -> A
    descend(api, bx, by, bz, br, Z_PLACE_OFFS)
    time.sleep(SETTLE_DWELL); grip(api, True); time.sleep(SETTLE_DWELL)
    ascend(api, bx, by, bz, br)

    descend(api, ax, ay, az, ar, Z_PICK_OFFS)
    time.sleep(SETTLE_DWELL); grip(api, False); time.sleep(SETTLE_DWELL)
    ascend(api, ax, ay, az, ar)

    ptp(api, *SAFE)
    input("[ENTER] — следующий слот…")

def main():
    api = connect()
    try:
        print("[INFO] HOME…"); home(api)
        print("[INFO] SAFE"); ptp(api, *SAFE)

        for slot in [A1, A2, A3]:
            print(f"\n=== {slot} ===")
            cycle_slot(api, slot, B)

        print("\n[INFO] Done. Back to SAFE")
        set_speed(api, *SPEED_TRANSIT); ptp(api, *SAFE)

    finally:
        stop_queue(api); dType.DisconnectDobot(api)

if __name__ == "__main__":
    main()
