import os
import pybullet as p
import pybullet_data
import numpy as np
import math

class TwoWheelRobot:
    def __init__(self,
                 render_mode='GUI',
                 enable_keyboard = False,
                 environment_type = 'FLAT',
                 time_step_size = 1/240,
                 x_distance_to_goal=10,
                 distance_to_goal_penalty=0.5,
                 robot_time_penalty=0.3,
                 target_velocity_change=0.5,
                 record_video = False,
                 multi_action = False,
                 goal_type = 'distance'):
        """Initialise Two Wheel Robot Environment.
        
        Args:
            render_mode (str, optional): _description_. Defaults to 'GUI'.
            enable_keyboard (bool, optional): _description_. Defaults to False.
            environment_type (str, optional): _description_. Defaults to 'FLAT'.
            time_step_size (_type_, optional): _description_. Defaults to 1/240.
            x_distance_to_goal (int, optional): _description_. Defaults to 10.
            distance_to_goal_penalty (float, optional): _description_. Defaults to 0.5.
            robot_time_penalty (float, optional): _description_. Defaults to 0.3.
            target_velocity_change (float, optional): _description_. Defaults to 0.5.
            record_video (bool, optional): _description_. Defaults to False.
            multi_action (bool, optional): _description_. Defaults to False.
        """
        # choose pybullet render mode
        if render_mode =='GUI': # run with GUI
            p.connect(p.GUI)
        elif render_mode =='DIRECT': # run headlessly
            p.connect(p.DIRECT)
        
        # get environment parameters
        self.target_velocity_change = target_velocity_change
        self.x_distance_to_goal = x_distance_to_goal
        self.distance_to_goal_penalty = distance_to_goal_penalty
        self.robot_time_penalty = robot_time_penalty
        self.flat_env = environment_type
        self.third_person_view = 'side' #'front' 
        self.enable_keyboard = enable_keyboard
        self.record_video = False #record_video #record_video
        self.multi_action = multi_action
        self.goal_type = goal_type
        self.reset()
        
        # set time step of simulation 
        self.time_step_size = time_step_size
        # if manual keyboard control or recording video, switch to smaller timesteps, else training/testing will default to 1/240 time steps
        if self.enable_keyboard or self.record_video:
            p.setTimeStep(24000)
                
        self.action_dict={
            0: "reverse_4",
            1: "reverse_3",
            2: "reverse_2",
            3: "reverse_1",
            4: "neutral",
            5: "forward_1",
            6: "forward_2",
            7: "forward_3",
            8: "forward_4"
        }

    def reset(self):
        """Resets Robot Environment after to prepare for a new episode run.
        """

        # resets the PyBullet Environment to remove all objects
        p.resetSimulation()
        
        # Sets Camera View for debug camera
        # sets third_person_view to face robot's front view
        if self.third_person_view == 'front':
            p.resetDebugVisualizerCamera(cameraDistance=3,  cameraYaw=-90, cameraPitch=-30, cameraTargetPosition=[0,0,1])
        # sets third_person_view to face robot's side view
        elif self.third_person_view == 'side':
            p.resetDebugVisualizerCamera(cameraDistance=3,  cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,1])
        # sets view to deafult view
        else:
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=-90, cameraPitch=-40, cameraTargetPosition=[5,0,0])

        # Load Environment 
        urdfRootPath = pybullet_data.getDataPath()
        self.planeid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,0])

        startOrientation = p.getQuaternionFromEuler([0,0,1.5707963])
        self.robotid = p.loadURDF("object_models/two_wheel_robot.xml",
                                    globalScaling=4.0, basePosition=[0,0,0], 
                                    baseOrientation=startOrientation,
                                    flags=p.URDF_USE_IMPLICIT_CYLINDER)
        
        # reset is called once at initialization of simulation
        self.target_velocity_left_wheel = 0 # initialise current target velocity of left wheel
        self.target_velocity_right_wheel = 0 # initialise current target velocity of right wheel
        self.maxV = 6.15 # max angular velocity in rad/s

        # set environment boundaries
        self.min_x_bound = -5
        self.max_x_bound = 45
        self.goal_line_x = self.x_distance_to_goal #self.max_x_bound - 35
        self.min_y_bound = -15
        self.max_y_bound = 15
        
        # set environment type (FLAT or SLOPE)
        if self.flat_env == 'SLOPE':
            self.load_slope()
            self.load_river()
        
        # Load environment boundaries
        self.load_boundaries()

        # Get robot dynamics
        self.robotdynamics = p.getDynamicsInfo(self.robotid,linkIndex=-1)
        self.numrobotJoints = p.getNumJoints(self.robotid)
        self.robot_base_position, self.robot_base_orientation = p.getBasePositionAndOrientation(self.robotid)

        # get robot links to name
        self._link_name_to_index = {p.getBodyInfo(self.robotid,)[0].decode('UTF-8'):-1,}
        for _id in range(p.getNumJoints(self.robotid,)):
            _name = p.getJointInfo(self.robotid, _id)[12].decode('UTF-8')
            self._link_name_to_index[_name] = _id

        # disable velocity motor for all robot joints
        p.setJointMotorControlArray(self.robotid, range(self.numrobotJoints), controlMode=p.VELOCITY_CONTROL, forces = [0 for _ in range(self.numrobotJoints)])
        
        # Set the friction coefficient for the front and rear wheel contact
        friction_coefficient = 1
        p.changeDynamics(self.robotid, self._link_name_to_index['left_wheel'], lateralFriction=friction_coefficient)
        p.changeDynamics(self.robotid, self._link_name_to_index['right_wheel'], lateralFriction=friction_coefficient)

        # initialise torque force of wheel
        self.force = 25
    
    def step(self, action, model_name, time_step, goal_step):
        """
        """
        # enable third person view, camera to follow robot
        if self.third_person_view == 'front':
            self.basePos, self.baseOrn = p.getBasePositionAndOrientation(self.robotid) # Get model position
            p.resetDebugVisualizerCamera( cameraDistance=3,  cameraYaw=-90, cameraPitch=-30, cameraTargetPosition=self.basePos) # fix camera onto model
        if self.third_person_view == 'side':
            self.basePos, self.baseOrn = p.getBasePositionAndOrientation(self.robotid) # Get model position
            p.resetDebugVisualizerCamera(cameraDistance=3,  cameraYaw=0, cameraPitch=-30, cameraTargetPosition=self.basePos)
        
        """ actuate action on physical robot in pybullet simulation """
        self.control_robot(action, model_name)

        # take steps in pybullet default time_step, apply robot actions based on user-defined time step size
        pybullet_steps = 24000 if (self.enable_keyboard or self.record_video) else 240 # 240 steps per second is pybullet default
        num_pybullet_steps_to_step_through = max(int(self.time_step_size * pybullet_steps),1)
        for _ in range(num_pybullet_steps_to_step_through):
            p.stepSimulation() # rum simulation for default time step of 1/240 seconds
            
        
        """ get observations from environment """
        observations = self.get_robot_state()
        
        """ check if agent is in terminal state after taking action """
        (done,succeed) = self.check_terminal_state(time_step=time_step, goal_step=goal_step) #end_episode == done
            
        # get state rewards rewards
        reward = self.return_state_reward(done,succeed) #reward
        
        # get transient rewards
        transient_reward = reward
        # distance goal type 
        if self.goal_type == 'distance':
            transient_time_reward = self.get_robot_time_reward(time_step=time_step,goal_steps=goal_step)
            transient_reward += transient_time_reward
            transient_distance_reward = self.get_shaped_reward_distance_to_goal(done)
            transient_reward += transient_distance_reward
        # time goal type
        elif self.goal_type == 'time':
            transient_time_reward = self.get_robot_time_reward(time_step=time_step,goal_steps=goal_step)
            transient_reward += transient_time_reward
            transient_velocity_reward = self.get_velocity_time_reward()
            transient_reward += transient_velocity_reward

        # print('transient reward:',round(transient_time_reward,10), round(transient_shape_reward,2),round(transient_reward,2))

        return observations, reward, transient_reward, done, succeed, {}

    def get_robot_state(self) -> np.array:
        """Get Robot State from the pybullet environment 

        Returns:
            self.observation (np.array): returns a (40, ) shape array. 
                (base_x, base_y, base_z, 
                base_orient_x, base_orient_y, base_orient_z,
                base_Vx, base_Vy, base_Vz,
                base_Wx, base_Wy, base_Wz,
                target_velocity_left_wheel, target_velocity_left_wheel,

                left_wheel_joint_angular_velocity,
                left_wheel_x, left_wheel_y, left_wheel_z, 
                left_wheel_orient_x, left_wheel_orient_y, left_wheel_orient_z, 
                left_wheel_Vx, left_wheel_Vy, left_wheel_Vz,
                left_wheel_Wx, left_wheel_Wy, left_wheel_Wz,

                right_wheel_joint_angular_velocity,
                right_wheel_x, right_wheel_y, right_wheel_z, 
                right_wheel_orient_x, right_wheel_orient_y, right_wheel_orient_z, 
                right_wheel_Vx, right_wheel_Vy, right_wheel_Vz,
                right_wheel_Wx, right_wheel_Wy, right_wheel_Wz,
                )
        """
        # assign current robot base position to previous robot_base position for calculation of distance reward shape function
        self.prev_robot_base_position = self.robot_base_position
        
        # get observations from robot base
        self.robot_base_position, self.robot_base_orientation = p.getBasePositionAndOrientation(self.robotid)
       
        # size 3, base linear velocity [x,y,z] in Cartesian world coordinates
        # size 3, base angular velocity [wx,wy,wz] in Cartesian world coordinates.
        base_linear_velocity, base_angular_velocity = p.getBaseVelocity(self.robotid)
        
        # size 3, robot_base_orientation: Wx, Wy, Wz in Cartesian world coordinates.
        robot_base_orientation_Euler = p.getEulerFromQuaternion(self.robot_base_orientation)

        # add robot base observations
        self.observation = np.concatenate((self.robot_base_position, np.array(robot_base_orientation_Euler), base_linear_velocity, base_angular_velocity), axis = None)

        # add robot target wheel angular velocities observations, size 2
        self.observation = np.concatenate((self.observation, self.target_velocity_left_wheel, self.target_velocity_right_wheel), axis = None)
        
        # get observations from robot joints and links
        link_states = list(p.getLinkStates(self.robotid, list(range(self.numrobotJoints)), computeLinkVelocity = 1)) 
        joint_states = list(p.getJointStates(self.robotid,list(range(self.numrobotJoints))))

        # loop through all robot joints and links (2 wheel joints)
        for joint_id in range(self.numrobotJoints):
            
            # size = 1, obtain angular velocity value of this wheel joint, w
            wheel_joint_angular_velocity = joint_states[joint_id][1]
            
            # size = 3, obtain local COM/center of mass (x,y,z) position offset of inertial frame expressed in URDF link frame
            local_link_com_position = np.array(link_states[joint_id][2]) # localInertialFramePosition
            
            # size = 3, roll around X, then pitch around Y and finally yaw around Z, as in the ROS URDF rpy convention
            # obtain local orientation (X,Y,Z) offset of the inertial frame expressed in URDF link frame.
            local_link_com_orientation = np.array(p.getEulerFromQuaternion(link_states[joint_id][3])) # localInertialFrameOrientation
            
            # size = 3, obtain Cartesian world velocity (Vx, Vy, Vz). Only returned if computeLinkVelocity non-zero.
            local_link_com_linear_velocity = np.array(link_states[joint_id][6]) # worldLinkLinearVelocity
            
            # size = 3, obtain Cartesian world velocity (Wx, Wy, Wz). Only returned if computeLinkVelocity non-zero.
            local_link_com_angular_velocity = np.array(link_states[joint_id][7]) # worldLinkAngularVelocity
            
            # concatenate and flatten observations
            self.observation = np.concatenate((self.observation,
                                                wheel_joint_angular_velocity,
                                                local_link_com_position,
                                                local_link_com_orientation,
                                                local_link_com_linear_velocity,
                                                local_link_com_angular_velocity), axis = None)

        # print('num of observations:',len(self.observation))                          

        return self.observation

    def normalize_observations(self, observations):
        """Normalize Observations to be used as inputs for the models

        Args:
            observations (list): state observations of TWR

        Returns:
            observations (list): normalized state observations of TWR
        """
        # robot body pitch angle, vt, Vx, Vy, Vz , Wx, Wy, Wz
        def normalize(value, min_value, max_value):
            return (value - min_value) / (max_value - min_value)

        # normalise base position, (x,y,z)
        min_value, max_value = -40, 40
        observations[0] = normalize(observations[0], min_value, max_value)
        observations[1] = normalize(observations[1], min_value, max_value)
        observations[2] = normalize(observations[2], min_value, max_value)

        # normalise base orientations, (x,y,z)
        min_value, max_value = -math.pi/4, math.pi/4 # pitch normalized to between -45 to 45 deg
        observations[3] = normalize(observations[3], min_value, max_value) # pitch
        min_value, max_value = -math.pi, math.pi
        observations[4] = normalize(observations[4], min_value, max_value) # roll
        min_value, max_value = -math.pi, 3*math.pi # starting position at +ve 90
        observations[5] = normalize(observations[5], min_value, max_value) # yaw

        # normalise base linear velocity, (Vx, Vy, Vz), v = r *w
        wheel_radius = 0.4
        min_value, max_value = -self.maxV * wheel_radius, self.maxV * wheel_radius
        observations[6] = normalize(observations[6], min_value, max_value)
        observations[7] = normalize(observations[7], min_value, max_value)
        observations[8] = normalize(observations[8], min_value, max_value)

        # normalise base angular velocity, (Wx, Wy, Wz)
        min_value, max_value = -self.maxV, self.maxV
        observations[9] = normalize(observations[9], min_value, max_value)
        observations[10] = normalize(observations[10], min_value, max_value)
        observations[11] = normalize(observations[11], min_value, max_value)
        
        # normalise target rotational velocity of left and right wheel, w
        min_value, max_value = -self.maxV, self.maxV
        observations[12] = normalize(observations[12], min_value, max_value) # left wheel
        observations[13] = normalize(observations[13], min_value, max_value) # right wheel

        """ normalize observations for left wheel """
        # normalize left wheel joint angular velocity, w
        wheel_radius = 0.4
        min_value, max_value = -self.maxV * wheel_radius, self.maxV * wheel_radius
        observations[14] = normalize(observations[14], min_value, max_value)
        
        # normalize left wheel COM position, (x,y,z)
        min_value, max_value = -2, 2
        observations[15] = normalize(observations[15], min_value, max_value)
        observations[16] = normalize(observations[16], min_value, max_value)
        observations[17] = normalize(observations[17], min_value, max_value)

        # normalize left wheel COM orientation, (x,y,z)
        min_value, max_value = -math.pi, math.pi
        observations[18] = normalize(observations[18], min_value, max_value)
        observations[19] = normalize(observations[19], min_value, max_value)
        observations[20] = normalize(observations[20], min_value, max_value)

        # normalise left wheel COM linear velocity, (Vx, Vy, Vz), v = r *w
        wheel_radius = 0.4
        min_value, max_value = -self.maxV * wheel_radius, self.maxV * wheel_radius
        observations[21] = normalize(observations[21], min_value, max_value)
        observations[22] = normalize(observations[22], min_value, max_value)
        observations[23] = normalize(observations[23], min_value, max_value)

         # normalise left wheel COM angular velocity, (Wx, Wy, Wz)
        min_value, max_value = -self.maxV, self.maxV
        observations[24] = normalize(observations[24], min_value, max_value)
        observations[25] = normalize(observations[25], min_value, max_value)
        observations[26] = normalize(observations[26], min_value, max_value)

        """ normalize observations for right wheel """
        # normalize right wheel joint angular velocity, w
        wheel_radius = 0.4
        min_value, max_value = -self.maxV * wheel_radius, self.maxV * wheel_radius
        observations[27] = normalize(observations[27], min_value, max_value)
        
        # normalize right wheel COM position, (x,y,z)
        min_value, max_value = -2, 2
        observations[28] = normalize(observations[28], min_value, max_value)
        observations[29] = normalize(observations[29], min_value, max_value)
        observations[30] = normalize(observations[30], min_value, max_value)

        # normalize right wheel COM orientation, (x,y,z)
        min_value, max_value = -math.pi, math.pi
        observations[31] = normalize(observations[31], min_value, max_value)
        observations[32] = normalize(observations[32], min_value, max_value)
        observations[33] = normalize(observations[33], min_value, max_value)

        # normalise right wheel COM linear velocity, (Vx, Vy, Vz), v = r *w
        wheel_radius = 0.4
        min_value, max_value = -self.maxV * wheel_radius, self.maxV * wheel_radius
        observations[34] = normalize(observations[34], min_value, max_value)
        observations[35] = normalize(observations[35], min_value, max_value)
        observations[36] = normalize(observations[36], min_value, max_value)

         # normalise right wheel COM angular velocity, (Wx, Wy, Wz)
        min_value, max_value = -self.maxV, self.maxV
        observations[37] = normalize(observations[37], min_value, max_value)
        observations[38] = normalize(observations[38], min_value, max_value)
        observations[39] = normalize(observations[39], min_value, max_value)
        # print('observations normalized:',observations)
        return observations
        
    def check_terminal_state(self, time_step, goal_step):
        """Check if robot has reached terminal state.

        Returns:
            bool0 (bool): Is terminal state reached?
            bool1 (bool): Is goal reached?
        """

        fail = (True,False)
        succeed = (True,True)
        ongoing = (False,False)

        # get contact points between robot and environment
        numContactPoints=0
        # if robot's body frame/handlebar is in contact with floor
        numContactPoints += len(p.getContactPoints(bodyA=self.robotid, linkIndexA=self._link_name_to_index['body'], bodyB=self.planeid, linkIndexB=-1))
        if self.flat_env == 'SLOPE':
            # if robot's body frame/handlebar is in contact with slope
            numContactPoints += len(p.getContactPoints(bodyA=self.robotid, linkIndexA=self._link_name_to_index['body'], bodyB=self.slopeid))
            # if robot touches river
            numContactPoints += len(p.getContactPoints(bodyA=self.robotid, bodyB=self.riverid))

        # if num contact points more than zero, terminal state reached
        if numContactPoints > 0:
            return fail

        # if robot's COM coordinates is outside defined boundaries, robot is in terminal state
        robotPos, _ = p.getBasePositionAndOrientation(self.robotid)
        x, y = robotPos[0], robotPos[1]
  
        
        # Based on how the robot urdf was created, we defined roll to be the pitch (and vice versa) in the robot urdf link
        pitch, roll, yaw = p.getEulerFromQuaternion(self.robot_base_orientation) # radians

        if abs(pitch* 180/math.pi) > 45: # if pitch is more than +- 45 degree
            return fail

        # set the goal type
        # if goal type is distance goal
        if self.goal_type == 'distance':
            #if robot's COM reaches the other side of riverbank and crosses the goal line, the robot is in terminal state (reached goal)
            if (x > self.goal_line_x) and (x < self.max_x_bound):
                return succeed
            # load boundaries to prevent robot from moving too far
            if x < self.min_x_bound or x > self.max_x_bound:
                return fail
            elif y < self.min_y_bound or y > self.max_y_bound:
                return fail
            
        # if goal type is time goal
        elif self.goal_type == 'time':
            if time_step > goal_step:
                return succeed
        else:
            raise ValueError("Goal Type not defined")

        # if all checks are okay, robot is in non-terminal state
        return ongoing
    
    def control_robot(self,action, model_name):
        """Control robot by action given by model/user

        Args:
            action (int): action giving by model/user
        """

        if self.enable_keyboard:
           action=self.action_dict[action]
        elif model_name == 'SAC': # continous
            action=action
        elif model_name == "DQN":
           action=self.action_dict[action.item()]
        elif model_name == "MAA2C" or model_name == "DQNMA":
           action=[self.action_dict[int(act)] for act in action]
        else:
            action=self.action_dict[action]
        
        
        def limit(value, min_value, max_value):
            """ Limits the value between min and max values. 
            If value is above max, returned value is max.
            If value is below min, returned value is min.
            If value is between min and max, returned value is value.

            Args:
                value (float): value to be processed
                min_value (float): minimum value
                max_value (float): maximum value

            Returns:
                processed_value: processed value
            """
            processed_value = max(min(value, max_value), min_value)
            return processed_value

        def get_change_in_wheel_velocity(action, dv):
            """ Get the Change in Wheel Velocity based on action

            Args:
                action (string): action giving by model/user
                dv (float): change in wheel angular velocity factor

            Returns:
                delta: change in wheel angular velocity in rad/s
            """
            deltav = 0
            if action == "forward_4":
                deltav = 5.*dv
            elif action == "forward_3":
                deltav = 3.*dv
            elif action == "forward_2":
                deltav = 1.*dv
            elif action == "forward_1":
                deltav = 0.1*dv
            elif action == "neutral":
                deltav = 0
            elif action == "reverse_1":
                deltav = -0.1*dv
            elif action == "reverse_2":
                deltav = -1.*dv
            elif action == "reverse_3":
                deltav = -3.*dv
            elif action == "reverse_4":
                deltav = -5.*dv
            else:
                print(f'action ({action}) is invalid!')
            return deltav 
        
        # get new target velocity of each wheel based on multi agent actions
        if model_name == 'MAA2C' or model_name == "DQNMA":
            self.target_velocity_left_wheel = limit(self.target_velocity_left_wheel + get_change_in_wheel_velocity(action[0], self.target_velocity_change), -self.maxV, self.maxV)
            self.target_velocity_right_wheel = limit(self.target_velocity_right_wheel + get_change_in_wheel_velocity(action[1], self.target_velocity_change), -self.maxV, self.maxV)
        elif model_name == 'SAC':
            self.target_velocity_left_wheel = limit(action, -self.maxV, self.maxV)
            self.target_velocity_right_wheel = limit(action, -self.maxV, self.maxV)
        # get new target velocity of each wheel based on single action
        else:
            self.target_velocity_left_wheel = limit(self.target_velocity_left_wheel + get_change_in_wheel_velocity(action, self.target_velocity_change), -self.maxV, self.maxV)
            self.target_velocity_right_wheel = limit(self.target_velocity_right_wheel  + get_change_in_wheel_velocity(action, self.target_velocity_change), -self.maxV, self.maxV)

        # apply new target velocity to left wheel joint
        p.setJointMotorControl2(bodyUniqueId=self.robotid, 
                                jointIndex=0, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=-self.target_velocity_left_wheel
                                )
        # apply new target velocity to right wheel joint
        p.setJointMotorControl2(bodyUniqueId=self.robotid, 
                                jointIndex=1, 
                                controlMode=p.VELOCITY_CONTROL, 
                                targetVelocity=self.target_velocity_right_wheel
                                )

    def get_action_space(self):
        """Get action space size 
        """
        return len(self.action_dict)
    
    """ ===== Methods for reward ===== """
    
    def return_state_reward(self, end_episode, succeed):
        """Return State Rewards

        Args:
            end_episode (bool): Boolean to check if episode has ended.
            succeed (bool): Boolean to check if robot has reached goal state.

        Returns:
            reward (int): reward value based on terminal state reached by robot.
        """
        
        reward_map = {"fall":-1,
                      "reached_goal":1,
                      "neutral":0}
        if end_episode:
            if succeed:
                return reward_map["reached_goal"]
            else:
                return reward_map["fall"]
        else:
            return reward_map["neutral"]
                
    def get_shaped_reward_distance_to_goal(self,end_episode):
        """Get shaped reward for distance to goal.

        Args:
            end_episode (bool): check if episode has ended

        Returns:
            transient_reward (float): transient_reward based on distance to goal.
        """
        temp_reward = 0
        if not end_episode:
            max_x_distance = self.goal_line_x - 0 # hardcorded 0 for starting position
            current_x = self.robot_base_position[0]
            prev_x = self.prev_robot_base_position[0]
            current_x_distance = self.goal_line_x - current_x # current distance to goal

            # if current state is closer to the goal line as compared to previous state, 
            # reward the agent for moving closer to goal
            if current_x > prev_x:
                # increasing linear transient reward gradient moving towards goal line.
                # closer the agent to goal, positive transient reward is larger. 
                # we normalize this transient reward based on the max_x_distance to the goal line
                temp_reward = self.distance_to_goal_penalty * (1 - (current_x_distance/(max_x_distance+ abs(self.min_x_bound))))

            # if current state's distance remains the same or is further away to goal comapred to previous state, do not penalize agent
            else:
                # increasing linear transient reward gradient moving away from goal line. 
                # further the agent to goal, negative transient reward is larger. 
                # we normalize this transient reward based on the max_x_distance to the goal line
                # temp_reward = -self.distance_to_goal_penalty * current_x_distance/max_x_distance
                pass

        return temp_reward

    def get_robot_time_reward(self, time_step, goal_steps):
        """get robot time penalty for stabilizing.
        """
        
        temp_reward = 0
        # rewards agent for survival time. time taken to reached goal / termination
        temp_reward +=  self.robot_time_penalty * time_step/goal_steps
        return temp_reward
    
    def get_velocity_time_reward(self):
        """get the velocity reward for staying a low speeds for balancing
        """
        temp_reward = 0
        temp_reward += 0.1 - abs(self.target_velocity_left_wheel - 0) * 0.010
        temp_reward += 0.1 - abs(self.target_velocity_right_wheel - 0) * 0.010
        return temp_reward
        
    """ ===== Methods to check robot states ===== """
    
    def print_robot_dimensions(self):
        """Get Dimensions of Robot and it's links
        """
        boundaries = p.getAABB(self.robotid,-1)
        lwh = np.array(boundaries[1])-np.array(boundaries[0])
        print(lwh)
        
        boundaries = p.getAABB(self.robotid,0)
        lwh = np.array(boundaries[1])-np.array(boundaries[0])
        print("left wheel dimensions (x,y,z):",lwh)

        boundaries = p.getAABB(self.robotid,1)
        lwh = np.array(boundaries[1])-np.array(boundaries[0])
        print("right wheel dimensions (x,y,z):",lwh)

        # Initialize the AABB to None
        aabbMin, aabbMax = None, None

        # Iterate through the joint states to get the AABB of each link
        for linkIndex in self._link_name_to_index.values():
            linkAABBMin, linkAABBMax = p.getAABB(self.robotid, linkIndex)

            # If this is the first AABB, initialize the AABB
            if aabbMin is None:
                aabbMin, aabbMax = linkAABBMin, linkAABBMax
            else:
                # Expand the AABB to include the new link's AABB
                aabbMin = [min(aabbMin[i], linkAABBMin[i]) for i in range(3)]
                aabbMax = [max(aabbMax[i], linkAABBMax[i]) for i in range(3)]

        # Print the final AABB
        print("AABB min:", aabbMin)
        print("AABB max:", aabbMax)
        print("entire robot dimensions (x,y,z):",np.array(aabbMax)-np.array(aabbMin))
        print("\n")

    def select_manual_action_from_keyboard(self,keys):
        """_Select Manual Action from Keyboard input.

        Args:
            keys (_type_): _description_

        Returns:
            _type_: _description_
        """
        action = 4 # 'neutral
        if p.B3G_UP_ARROW in keys:
            action = 5 #"forward_1"
        if p.B3G_DOWN_ARROW in keys:
            action = 3 #"reverse_1"
        if ord('t') in keys:
            action = 6 #"forward_1"
        if ord('g') in keys:
            action = 2 
        return action

    def load_river(self):
        """Loads the river model
        """
        floorLength = 27.0
        floorWidth = abs(self.min_y_bound) + abs(self.max_y_bound)
        floorHeight = 0.01
        # create river
        riverid = p.createCollisionShape(p.GEOM_BOX, halfExtents=[floorLength/2, floorWidth/2, floorHeight/2])
        self.riverid = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=riverid, basePosition=[20,0,0])
        # set the color of the river to light blue
        p.changeVisualShape(self.riverid, -1, rgbaColor=[0, 1, 1, 1])

    def load_boundaries(self):
        """Loads the visualisation of the environment boundaries
        """
        # Define the dimensions of the environment boundaries
        edgeHeight = 0.1
        edgeHeight = edgeHeight / 2

        edgeWidth = 0.1
        edgeLength = abs(self.min_y_bound) + abs(self.max_y_bound)

        # render south boundary
        southEdgeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[edgeWidth/2, edgeLength/2, edgeHeight/2], rgbaColor=[1,0,0,1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=southEdgeId, basePosition=[self.min_x_bound, 0, 0])

        # render north boundary
        northEdgeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[edgeWidth/2, edgeLength/2, edgeHeight/2], rgbaColor=[1,0,0,1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=northEdgeId, basePosition=[self.max_x_bound, 0, 0])

        # render goal line
        goalEdgeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[edgeWidth/2, edgeLength/2, edgeHeight/2], rgbaColor=[0,1,0,1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=goalEdgeId, basePosition=[self.goal_line_x, 0, 0])

        edgeWidth = self.max_x_bound + 5
        edgeLength = 0.1
        # render east boundary
        eastEdgeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[edgeWidth/2, edgeLength/2, edgeHeight/2], rgbaColor=[1,0,0,1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=eastEdgeId, basePosition=[(self.max_x_bound-5)/2, self.min_y_bound, 0])
        # render west boundary
        westEdgeId = p.createVisualShape(p.GEOM_BOX, halfExtents=[edgeWidth/2, edgeLength/2, edgeHeight/2], rgbaColor=[1,0,0,1])
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,baseVisualShapeIndex=eastEdgeId, basePosition=[(self.max_x_bound-5)/2, self.max_y_bound, 0])
       
    def load_slope(self):
        """Loads the slope model
        """
        self.slopeid = p.loadURDF("object_models/slope_5deg.urdf",
                            [20,0,-0.5], 
                            [0,0,0,1], 
                            useFixedBase=True, 
                            globalScaling=2.5)

    def get_robot_joints(self):
        """Get Robot Joint Information
        """
        self.joint_dict = {}
        # iterate over each joint and get its information
        for i in range(self.numrobotJoints):
            jointInfo = p.getJointInfo(self.robotid, i)
            jointName = jointInfo[1].decode("utf-8")
            jointType = jointInfo[2]
            qIndex = jointInfo[3]
            uIndex = jointInfo[4]
            flags = jointInfo[5]
            jointDamping = jointInfo[6]
            jointFriction = jointInfo[7]
            jointLowerLimit = jointInfo[8]
            jointUpperLimit = jointInfo[9]
            jointMaxForce = jointInfo[10]
            jointMaxVelocity = jointInfo[11]
            linkName = jointInfo[12]
            print(f"Joint {i}: {jointName}, Type: {jointType}, Limits: ({jointLowerLimit}, {jointUpperLimit}), Max Force: {jointMaxForce}, Max Velocity: {jointMaxVelocity}")
            self.joint_dict[jointName] = {'index':i,
                                     'jointLowerLimit':jointLowerLimit,
                                     'jointUpperLimit':jointUpperLimit}
            
    def print_robot_attributes(self):
        """Prints robot Attributes
        """
        print("base (or root link) of the body : position , orientation",p.getBasePositionAndOrientation(self.robotid))
        print("roll around X, pitch around Y, yaw around Z", p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robotid)[1]))
        print('self._link_name_to_index:',self._link_name_to_index)
        print(f"  Mass: {self.robotdynamics[0]}")

    def close(self):
        """Close and Disconnect PyBullet Simulation Window
        """
        p.disconnect()

if __name__ == '__main__':
    robot_env = TwoWheelRobot(render_mode='GUI',
                    enable_keyboard=True)
