import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import ConvexHull
from skimage.transform import resize
from skimage import img_as_ubyte
import warnings
import torch
import yaml
from tqdm import tqdm

from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork


class Tps:
    def __init__(self):
        self.filename = 'winter'
        self.device = torch.device('cuda:0')
        self.dataset_name = 'vox'
        self.source_image_path = f'./data/{self.filename}_align.png'
        self.driving_video_path = './data/driving.mp4'
        self.output_video_path = f'./generated./{self.filename}.mp4'
        self.config_path = './config/vox-256.yaml'
        self.checkpoint_path = './checkpoints/vox.pth.tar'
        self.predict_mode = 'relative'
        self.find_best_frame = False
        self.pixel = 256

    def load_source(self):
        warnings.filterwarnings("ignore")
        source_image = imageio.imread(self.source_image_path)
        reader = imageio.get_reader(self.driving_video_path)
        self.source_image = resize(source_image, (self.pixel, self.pixel))[..., :3]

        self.fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        self.driving_video = [resize(frame, (self.pixel, self.pixel))[..., :3] for frame in driving_video]

    def load_checkpoints(self):
        device = self.device
        with open(self.config_path) as f:
            config = yaml.full_load(f)

        inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                       **config['model_params']['common_params'])
        kp_detector = KPDetector(**config['model_params']['common_params'])
        dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                                  **config['model_params']['dense_motion_params'])
        avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                                 **config['model_params']['avd_network_params'])
        kp_detector.to(device)
        dense_motion_network.to(device)
        inpainting.to(device)
        avd_network.to(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=device)

        inpainting.load_state_dict(checkpoint['inpainting_network'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
        if 'avd_network' in checkpoint:
            avd_network.load_state_dict(checkpoint['avd_network'])

        inpainting.eval()
        kp_detector.eval()
        dense_motion_network.eval()
        avd_network.eval()

        return inpainting, kp_detector, dense_motion_network, avd_network

    def relative_kp(self, kp_source, kp_driving, kp_driving_initial):

        source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

        kp_new = {k: v for k, v in kp_driving.items()}

        kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
        kp_value_diff *= adapt_movement_scale
        kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

        return kp_new

    def make_animation(self, source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, avd_network,
                       device, mode='relative'):
        assert mode in ['standard', 'relative', 'avd']
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            source = source.to(device)
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(
                device)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                driving_frame = driving_frame.to(device)
                kp_driving = kp_detector(driving_frame)
                if mode == 'standard':
                    kp_norm = kp_driving
                elif mode == 'relative':
                    kp_norm = self.relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                          kp_driving_initial=kp_driving_initial)
                elif mode == 'avd':
                    kp_norm = avd_network(kp_source, kp_driving)
                dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param=None,
                                                    dropout_flag=False)
                out = inpainting_network(source, dense_motion)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions

    def make_video(self):
        warnings.filterwarnings("ignore")
        source_image = imageio.imread(self.source_image_path)
        reader = imageio.get_reader(self.driving_video_path)
        source_image = resize(source_image, (self.pixel, self.pixel))[..., :3]

        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        driving_video = [resize(frame, (self.pixel, self.pixel))[..., :3] for frame in driving_video]

        inpainting, kp_detector, dense_motion_network, avd_network = self.load_checkpoints()
        device = self.device
        predict_mode = self.predict_mode

        if predict_mode == 'relative' and self.find_best_frame:
            from demo import find_best_frame as _find
            i = _find(source_image, driving_video, self.device.type == 'cpu')
            print("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i + 1)][::-1]
            predictions_forward = self.make_animation(source_image, driving_forward, inpainting, kp_detector,
                                                 dense_motion_network, avd_network, device=device, mode=predict_mode)
            predictions_backward = self.make_animation(source_image, driving_backward, inpainting, kp_detector,
                                                  dense_motion_network, avd_network, device=device, mode=predict_mode)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = self.make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network,
                                         avd_network, device=device, mode=predict_mode)

        # save resulting video
        imageio.mimsave(self.output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)


if __name__ == '__main__':
    Tps().make_video()