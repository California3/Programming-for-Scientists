import robot

def pickup_box():
    robot.gripper_to_open()
    robot.lift_down()
    robot.gripper_to_closed()
    robot.lift_up()

def twice_right():
    robot.drive_right()
    robot.drive_right()


robot.init()

robot.drive_right()
robot.lift_up()
pickup_box()

twice_right()

pickup_box()

twice_right()
robot.gripper_to_folded()
robot.lift_down()