import RPi.GPIO as GPIO
import time
from AlphaBot import AlphaBot

Ab = AlphaBot()

#IR = 18
#PWM = 50
#n=0

#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)
#GPIO.setup(IR,GPIO.IN,GPIO.PUD_UP)
print('IRremote Test Start ...')
#Ab.stop()
key=int(input())

while key!=0 :
    if key == 1:
        Ab.forward()
        print("forward")
    if key == 2:
        Ab.stop()
        print("stop")
    if key == 3:
        Ab.left()
        print("left")
    if key == 4:
        Ab.right()
        print("right")
    if key == 5:
        Ab.backward()
        print("backward")
    if key == 6:
        GPIO.cleanup();
        break
    if key == 7:
        if(PWM + 10 < 101):
            PWM = PWM + 10
            Ab.setPWMA(PWM)
            Ab.setPWMB(PWM)
            print(PWM)
    if key == 8:
        if(PWM - 10 > -1):
            PWM = PWM - 10
            Ab.setPWMA(PWM)
            Ab.setPWMB(PWM)
            print(PWM)
            
    key = int(input())

