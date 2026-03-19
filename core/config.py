import numpy as np

SIMULATION_DT = 0.01
TOTAL_SIMULATION_TIME = 140.0
PLAYBACK_SPEED = 6.0


WORLD_SIZE = [980, 889]
PIXELS_PER_METER = 130

COLLISION_DETECTION = True
FRICTION_COEFFICIENT = 0.1

IMU_PARAMS = {
    'frequency': 100,
    'noise_std': 0.01,
    'accel_bias_range': 0.01,
    'gyro_bias_range': 0.001,
    'accel_scale_range': 0.01,
    'gyro_scale_range': 0.01
}

UWB_PARAMS = {
    'frequency': 10,
    'noise_std': 0.1,
    'range_limit': 5000.0,
    'nlos_probability': 0.1,
    'nlos_bias': 0.5
}

NUM_AGENTS = 3
AGENTS = [
    {
        'id': 1,
        'type': 'differential',
        'initial_position': [35, 35, 0],
        'controller': 'user_scripts.agent_1_controller.Agent1Controller',
        'sensors': {
            'imu': IMU_PARAMS,
            'uwb': UWB_PARAMS
        }
    },
    {
        'id': 2,
        'type': 'differential',
        'initial_position': [944, 844, 0],
        'controller': 'user_scripts.agent_2_controller.Agent2Controller',
        'sensors': {
            'imu': IMU_PARAMS,
            'uwb': UWB_PARAMS
        }
    },
    {
        'id': 3,
        'type': 'differential',
        'initial_position': [35, 844, 0],
        'controller': 'user_scripts.agent_3_controller.Agent3Controller',
        'sensors': {
            'imu': IMU_PARAMS,
            'uwb': UWB_PARAMS
        }
    }
]

UWB_ANCHORS = [
    [0, 0, 1.5],
    [980, 0, 1.5],
    [0, 890, 1.5],
    [980, 890, 1.5]
]

WAYPOINTS = [
    {'position': [945, 32], 'tag': '1'},
    {'position': [945, 665], 'tag': '2'},
    {'position': [945, 852], 'tag': '3'},
    {'position': [618, 854], 'tag': '4'},
    {'position': [458, 854], 'tag': '5'},
    {'position': [31, 855], 'tag': '6'},
    {'position': [31, 532], 'tag': '7'},
    {'position': [31, 272], 'tag': '8'},
    {'position': [31, 31], 'tag': '9'},
    {'position': [316, 31], 'tag': '10'},
    {'position': [719, 539], 'tag': '11'},
    {'position': [770, 443], 'tag': '12'},
    {'position': [603, 343], 'tag': '13'},
    {'position': [457, 536], 'tag': '14'},
    {'position': [318, 247], 'tag': '15'},
]

Task = [945, 32]

OBSTACLES = [
    {'type': 'polygon', 'points': [[65, 65], [286, 65], [286, 241], [65, 241]]},
    {'type': 'polygon', 'points': [[65, 304], [288, 306], [420, 504], [67, 501]]},
    {'type': 'polygon', 'points': [[66, 570], [425, 572], [428, 822], [65, 824]]},
    {'type': 'polygon', 'points': [[352, 257], [521, 257], [559, 336], [527, 370], [507, 418], [506, 471]]},
    {'type': 'polygon', 'points': [[506, 471], [519, 516], [542, 548], [586, 582], [586, 822], [493, 822], [493, 493], [352, 282], [352, 257]]},
    {'type': 'polygon', 'points': [[618, 593], [643, 595], [670, 595], [704, 585], [618, 735]]},
    {'type': 'polygon', 'points': [[652, 823], [651, 743], [762, 553], [788, 541], [818, 554], [913, 699], [914, 823]]},
    {'type': 'polygon', 'points': [[352, 64], [913, 66], [913, 583], [759, 351], [718, 320], [668, 304], [620, 308], [563, 191], [352, 191]]}
]

BARRIERS = [
    {'start': [268, 503], 'end': [268, 570]},
    {'start': [874, 478], 'end': [832, 600]},
    {'start': [423, 702], 'end': [494, 702]}
]
