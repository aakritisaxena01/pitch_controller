#with pid controller
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pymavlink import mavutil
import pyrealsense2 as rs
# ---------------------------
# CONFIG
# ---------------------------
TARGET_CLASS_ID = 0          # 0 = person, change as needed
THRESHOLD_PX    = 25.0      
YAW_GAIN        = 0.1       
PITCH_PGAIN     = 0.18
PITCH_IGAIN     = 0.0022
PITCH_DGAIN     = 0.0038
SERVO_RANGE_DEG = 90
SERVO_NEUTRAL   = 45
model = YOLO("yolo11n.pt")
#cap   = cv2.VideoCapture(1)  
master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
master.wait_heartbeat()
print("Connected")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)
for i in range(30):
    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    print(f"Frame {i}: color valid = {bool(color)}")  # tells you whats actually arriving
print("Camera ready")

'''mode_id = master.mode_mapping()['GUIDED']
master.set_mode(mode_id)
time.sleep(2)

# Arm
master.arducopter_arm()
master.motors_armed_wait()
print("Armed")

# Takeoff
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, 0, 0, 0, 0, 0, 0, 10
)
time.sleep(8)'''

# ---------------------------
# YAW CONTROL FUNCTION
# ---------------------------
def set_yaw(angle_deg):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        abs(angle_deg),
        10,
        1 if angle_deg > 0 else -1,
        1,
        0, 0, 0
    )

def angle_to_pwm(angle_deg):
    angle_deg = max(0, min(SERVO_RANGE_DEG, angle_deg))
    return int(500 + (angle_deg / SERVO_RANGE_DEG) * 2000)

def set_servo_pwm(channel: int, pwm_us: int):
    pwm_us = max(500, min(2500, pwm_us))   # clamp to safe range
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,   # command id 183
        0,          # confirmation
        channel,    # param1: servo channel number
        pwm_us,     # param2: PWM value (µs)
        0, 0, 0, 0, 0   # unused params
    )
    print(f"  → Channel {channel} set to {pwm_us} us")
integral = 0.0
prev_error_y = 0.0
pitch_angle = 45.0
INTEGRAL_CLAMP = 50.0   # anti-windup
'''if not cap.isOpened():
    print("Error: Could not open the external camera.")
    exit()'''
def pid(error_y, dt):
    global integral, prev_error_y
    p = error_y
    d = (error_y - prev_error_y) / dt
    integral += error_y * dt
    integral = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, integral))  
    prev_error_y = error_y
    print("integral term: ", PITCH_IGAIN * integral, "derivative term: ", PITCH_DGAIN * d, "proportional term: ", PITCH_PGAIN * p)
    return PITCH_PGAIN * p - PITCH_DGAIN * d + PITCH_IGAIN * integral

'''while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_cx = w // 2
    frame_cy = h // 2

    results = model(frame, verbose=False)'''
set_servo_pwm(9, 1500)
i=0  
start_time = time.time()
p_dt = time.time()
time.sleep(2)
pitch_angle = 45.0
while True:
    bestframe = None
    list1=[]
    list2=[]
    for i in range(5):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=10000) 
        except RuntimeError as e:
            print(f"Frame timeout, skipping: {e}")
            continue  
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())

        h, w, _ = frame.shape
        frame_cx = w // 2
        frame_cy = h // 2

        results = model(frame, verbose=False)

        best_box  = None
        best_area = 0
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != TARGET_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box  = (x1, y1, x2, y2)
            list1.append(best_area)
            list2.append(best_box)
    maxindex=list1.index(max(list1))
    best_frame=list2[maxindex]
    yaw_change = 0.0    
    error_x = 0
    error_y = 0
    
    if best_frame is not None:
        i=0
        x1, y1, x2, y2 = best_frame
        obj_cx = (x1 + x2) // 2
        obj_cy = (y1 + y2) // 2
        error_x = obj_cx - frame_cx   # +ve = object right of center
        error_y = obj_cy - frame_cy   # +ve = object below center
        
        if abs(error_x) > THRESHOLD_PX:
            yaw_change = abs(error_x) * YAW_GAIN
            #set_yaw(yaw_change)         
        MAX_ANGLE_CHANGE = 1.5  
        if abs(error_y) > THRESHOLD_PX:
            dt   = time.time() - p_dt
            p_dt = time.time()
            pidc = pid(error_y, dt)
            
            target_pitch = SERVO_NEUTRAL + pidc
            target_pitch = max(0, min(90, target_pitch))

            delta = target_pitch - pitch_angle
            delta = max(-MAX_ANGLE_CHANGE, min(MAX_ANGLE_CHANGE, delta))
            pitch_angle += delta                  # creep toward target

            pwm = angle_to_pwm(pitch_angle)
            set_servo_pwm(9, pwm)
            i=i+1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                print("number of command sent per sec", i)
                i=0
                start_time = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (obj_cx, obj_cy), 6, (0, 0, 255), -1)

        cv2.line(frame, (frame_cx, frame_cy), (obj_cx, frame_cy), (0, 255, 255), 1)  # horizontal error
        cv2.line(frame, (obj_cx, frame_cy), (obj_cx, obj_cy),     (0, 165, 255), 1)  # vertical error

        label = f"{model.names[TARGET_CLASS_ID]}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        integral     = 0.0
        prev_error_y = 0.0
        #pitch_angle  = float(SERVO_NEUTRAL)
        #set_servo_pwm(9, 1500)
    
    cv2.line(frame, (frame_cx, 0),      (frame_cx, h),      (255, 0, 0), 1)
    cv2.line(frame, (0, frame_cy),      (w, frame_cy),      (255, 0, 0), 1)
    cv2.circle(frame, (frame_cx, frame_cy), 5, (255, 0, 0), -1)

   
    pwm = angle_to_pwm(pitch_angle)
    cv2.putText(frame, f"Pitch angle : {pitch_angle:.1f} deg",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Pitch PWM   : {pwm} us",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Yaw change  : {yaw_change:.2f} deg",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Error X     : {int(error_x) if best_box else 0:+d} px",
                (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"Error Y     : {int(error_y) if best_box else 0:+d} px",
                (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    master.recv_match(blocking=False)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


pipeline.stop()
cv2.destroyAllWindows()
