import cv2
import numpy as np

def random_env_step(i, env, framework):
    action = env.action_space.no_op()

    if framework == 'minerl':
        action["ESC"] = 0           # avoid ending the episode early
        action['forward'] = 1       # move forward
        if i % 10 == 0:
            action['jump'] = 1      # jump
    elif framework == 'minedojo':
        action[0] = 1               # move forward 
        if i % 10 == 0:
            action[2] = 1           # jump

    obs, _, done, _ = env.step(action)
    if done:
        env.reset()
    
    return obs

def get_frames_from_observations(observations, framework):
    if framework == 'minerl':
        return [obs['pov'] for obs in observations]
    elif framework == 'minedojo':
        return [obs['rgb'] for obs in observations]
    else:
        raise ValueError(f"Invalid framework: {framework}")

def print_space(space, framework):
    dict_cls = _get_dict_class_from_framework(framework)
    
    _print_space_r(space, dict_cls)


def _print_space_r(obj, dict_cls, key=None, tabs=0):
    offset = ' ' * tabs * 4

    if key == None:
        print(offset, end='')
    else:
        print(offset, key, ': ', sep='', end='')

    if not isinstance(obj, dict_cls):
        print(obj, ',', sep='')
        return

    print('{', sep='')

    for key in obj:
        value = obj[key]

        _print_space_r(value, dict_cls, key=key, tabs=tabs + 1)

    print(offset, '},', sep='')

def _get_dict_class_from_framework(framework):
    if framework == 'minerl':
        import minerl
        dict_cls = minerl.herobraine.hero.spaces.Dict
    elif framework == 'minedojo':
        import gym
        dict_cls = gym.spaces.Dict
    else:
        raise ValueError(f"Invalid framework: {framework}")
    
    return dict_cls

def save_video(
        frames,
        filename,
        fps=20,
        channel_first=True,
        low=0.,
        high=1.,
        invert_rgb=False,
    ):
    if channel_first:
        frame_size = frames[0].shape[1:]
    else:
        frame_size = frames[0].shape[:2]

    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_size[1], frame_size[0])
    )

    for frame in frames:
        if channel_first:
            frame = np.transpose(frame, (1, 2, 0))

        frame = (frame - low) / (high - low) * 255.
        frame = frame.astype(np.uint8)

        if invert_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out.write(frame)

    out.release()