from hand_utils.hand_model import create_hand_model


if __name__ == "__main__":


    hand_models = [create_hand_model('shadowhand', torch.device('cpu'))]
    qpos = ...
    hand_mesh = hand_models[0].get_trimesh_q(qpos)['visual']
    