# For better visualization, give different colors to different bones

h36m_elbow_knee_v1 = [5, 15]
h36m_elbow_knee_v2 = [2, 12]
h36m_wrist_ankle_v1 = [6, 16]
h36m_wrist_ankle_v2 = [3, 13]
h36m_hip_shoulder = [1, 4, 11, 14]
h36m_spine_neck = [7, 9]
h36m_thorax_head = [8, 10]


def h36m_color_edge(joint_num):
    if joint_num in h36m_elbow_knee_v1:
        color = 'peru'  # (205, 133, 63)
    elif joint_num in h36m_elbow_knee_v2:
        color = 'indianred'  # (205, 92, 92)
    elif joint_num in h36m_wrist_ankle_v1:
        color = 'coral'  # (255, 127, 80)
    elif joint_num in h36m_wrist_ankle_v2:
        # color = 'deepskyblue'
        color = 'brown'  # (165, 42, 42)
    elif joint_num in h36m_hip_shoulder:
        # color = 'dodgerblue'
        color = 'tan'  # (210, 180, 140)
    elif joint_num in h36m_spine_neck:
        color = 'olive'  # (128, 128, 0)
    else:
        color = 'purple'  # (128, 0, 128)
    return color


ntu_elbow_knee_v1 = [6, 18]
ntu_elbow_knee_v2 = [10, 14]
ntu_wrist_ankle_v1 = [8, 19]
ntu_wrist_ankle_v2 = [12, 15]
ntu_hip_shoulder = [13, 17, 5, 9]
ntu_spine_neck = [2, 3]
ntu_thorax_head = [21, 4]
ntu_foot = [16, 20]
ntu_middle_wrist = [7, 11]
ntu_thumbs = [23, 25]
ntu_middle_finger = [22, 24]


def ntu_color_edge(joint_num):
    if joint_num in ntu_elbow_knee_v1:
        color = 'peru'
    elif joint_num in ntu_elbow_knee_v2:
        color = 'indianred'
    elif joint_num in ntu_wrist_ankle_v1:
        color = 'coral'
    elif joint_num in ntu_wrist_ankle_v2:
        color = 'brown'
    elif joint_num in ntu_hip_shoulder:
        color = 'tan'
    elif joint_num in ntu_spine_neck:
        color = 'olive'
    elif ntu_thorax_head:
        color = 'purple'
    elif joint_num in ntu_foot:
        color = 'deepskyblue'
    elif joint_num in ntu_middle_wrist:
        color = 'dodgerblue'
    elif joint_num in ntu_thumbs:
        color = 'red'
    else:
        color = 'yellow'
    return color
