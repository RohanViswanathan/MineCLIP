import torch
import hydra
from omegaconf import OmegaConf

from mineclip import MineCLIP
import numpy as np
from torch.utils.data import DataLoader

def mineclip_reward(frames: np.ndarray, task_prompt: list):
    print(type(frames))
    reward = None

    @torch.no_grad()
    @hydra.main(config_name="conf", config_path=".", version_base="1.1")
    def calculate_reward(cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OmegaConf.set_struct(cfg, False)
        cfg.pop("ckpt")
        OmegaConf.set_struct(cfg, True)

        model = MineCLIP(**cfg).to(device)

        BATCH_SIZE = 16

        total_frames = frames.shape[0]
        num_batches = total_frames // BATCH_SIZE

        trimmed_frames = frames[:num_batches * BATCH_SIZE]
        # reshaped_array = trimmed_frames.reshape(-1, 8, 3, 160, 256)

        # video = torch.from_numpy(reshaped_array)
        # video = video.to(device=device)

        # print("video shape:", video.shape)

        # prompts = task_prompt
        
        # VIDEO_BATCH, TEXT_BATCH = video.size(0), len(prompts)

        # image_feats = model.forward_image_features(video)
        # video_feats = model.forward_video_features(image_feats)
        # assert video_feats.shape == (VIDEO_BATCH, 512)

        reshaped_array = trimmed_frames.reshape(-1, BATCH_SIZE, 3, 160, 256)

        video = torch.from_numpy(reshaped_array).to(device=device)
        print("video shape:", video.shape)

        prompts = task_prompt

        video_loader = DataLoader(video, batch_size=BATCH_SIZE)

        cum_reward = torch.zeros([BATCH_SIZE, 2]).to(device)

        for video_batch in video_loader:
            VIDEO_BATCH = video_batch.size(0)
            TEXT_BATCH = len(prompts)

            image_feats = model.forward_image_features(video_batch)
            video_feats = model.forward_video_features(image_feats)
            assert video_feats.shape == (VIDEO_BATCH, 512)

        # video_feats_2 = model.encode_video(video)
        # # encode_video is equivalent to forward_video_features(forward_image_features(video))
        # torch.testing.assert_allclose(video_feats, video_feats_2)

        # # encode batch of prompts
        # text_feats_batch = model.encode_text(prompts)
        # assert text_feats_batch.shape == (TEXT_BATCH, 512)

        # # compute reward from features
        # logits_per_video, logits_per_text = model.forward_reward_head(
        #     video_feats, text_tokens=text_feats_batch
        # )
        # assert logits_per_video.shape == (VIDEO_BATCH, TEXT_BATCH)
        # assert logits_per_text.shape == (TEXT_BATCH, VIDEO_BATCH)
        # directly pass in strings. This invokes the tokenizer under the hood
            reward_scores_2, _ = model.forward_reward_head(video_feats, text_tokens=prompts)
            print("shapes of cum reward/reward", cum_reward.shape, reward_scores_2.shape)

            if reward_scores_2.shape == cum_reward:
                cum_reward += reward_scores_2
        # pass in cached, encoded text features
        # reward_scores_3, _ = model(
        #     video_feats, text_tokens=text_feats_batch, is_video_features=True
        # )
        # reward_scores_4, _ = model(
        #     video, text_tokens=text_feats_batch, is_video_features=False
        # )
        # all above are equivalent, just starting from features or raw values
        # torch.testing.assert_allclose(logits_per_video, reward_scores_2)
        # torch.testing.assert_allclose(logits_per_video, reward_scores_3)
        # torch.testing.assert_allclose(logits_per_video, reward_scores_4)

        cum_reward = cum_reward/num_batches

        nonlocal reward
        reward = reward_scores_2

    calculate_reward()
    
    return reward

if __name__ == "__main__":
    rew = mineclip_reward(frames=np.load('main/mineclip/frames_test.npy'))
    print(rew)