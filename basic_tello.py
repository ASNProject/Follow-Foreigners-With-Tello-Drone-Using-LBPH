from djitellopy import tello

me = tello.Tello()

me.connect()
me.takeoff()

me.rotate_counter_clockwise(90)

me.land()