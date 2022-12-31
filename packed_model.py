import os
import sys
from dataclasses import dataclass

import PIL.Image
from loguru import logger

logger.info("Importing libraries...")

device = "mps"

sys.path.append(".")

import numpy as np
import torch
import dlib
from util import load_image
from argparse import Namespace
from torchvision import transforms
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from model.encoder.align_all_parallel import align_face

logger.info("Libraries imported.")

MODEL_DIR = os.path.join(os.getcwd(), "checkpoint")
DATA_DIR = os.path.join(os.getcwd(), "data")
DLIB_FACE_MODEL_NAME = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')

style_types = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']
style_type = style_types[0]

if not os.path.exists(os.path.join(MODEL_DIR, style_type)):
    os.makedirs(os.path.join(MODEL_DIR, style_type))


def get_download_model_command(file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../checkpoint/. """
    # current_directory = os.getcwd()
    save_path = MODEL_DIR
    # skip download if exists
    if os.path.exists(os.path.join(save_path, file_name)):
        return "echo 'Model exists, skipping download.'"
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(
        FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url


MODEL_PATHS = {
    "encoder": {"id": "1NgI4mPkboYvYw3MWcdUaQhkr0OWgs9ej", "name": "encoder.pt"},
    "cartoon-G": {"id": "1exS9cSFkg8J4keKPmq2zYQYfJYC5FkwL", "name": "generator.pt"},
    "cartoon-N": {"id": "1JSCdO0hx8Z5mi5Q5hI9HMFhLQKykFX5N", "name": "sampler.pt"},
    "cartoon-S": {"id": "1ce9v69JyW_Dtf7NhbOkfpH77bS_RK0vB", "name": "refined_exstyle_code.npy"},
    "caricature-G": {"id": "1BXfTiMlvow7LR7w8w0cNfqIl-q2z0Hgc", "name": "generator.pt"},
    "caricature-N": {"id": "1eJSoaGD7X0VbHS47YLehZayhWDSZ4L2Q", "name": "sampler.pt"},
    "caricature-S": {"id": "1-p1FMRzP_msqkjndRK_0JasTdwQKDsov", "name": "refined_exstyle_code.npy"},
    "anime-G": {"id": "1BToWH-9kEZIx2r5yFkbjoMw0642usI6y", "name": "generator.pt"},
    "anime-N": {"id": "19rLqx_s_SUdiROGnF_C6_uOiINiNZ7g2", "name": "sampler.pt"},
    "anime-S": {"id": "17-f7KtrgaQcnZysAftPogeBwz5nOWYuM", "name": "refined_exstyle_code.npy"},
    "arcane-G": {"id": "15l2O7NOUAKXikZ96XpD-4khtbRtEAg-Q", "name": "generator.pt"},
    "arcane-N": {"id": "1fa7p9ZtzV8wcasPqCYWMVFpb4BatwQHg", "name": "sampler.pt"},
    "arcane-S": {"id": "1z3Nfbir5rN4CrzatfcgQ8u-x4V44QCn1", "name": "exstyle_code.npy"},
    "comic-G": {"id": "1_t8lf9lTJLnLXrzhm7kPTSuNDdiZnyqE", "name": "generator.pt"},
    "comic-N": {"id": "1RXrJPodIn7lCzdb5BFc03kKqHEazaJ-S", "name": "sampler.pt"},
    "comic-S": {"id": "1ZfQ5quFqijvK3hO6f-YDYJMqd-UuQtU-", "name": "exstyle_code.npy"},
    "pixar-G": {"id": "1TgH7WojxiJXQfnCroSRYc7BgxvYH9i81", "name": "generator.pt"},
    "pixar-N": {"id": "18e5AoQ8js4iuck7VgI3hM_caCX5lXlH_", "name": "sampler.pt"},
    "pixar-S": {"id": "1I9mRTX2QnadSDDJIYM_ntyLrXjZoN7L-", "name": "exstyle_code.npy"},
    "slamdunk-G": {"id": "1MGGxSCtyf9399squ3l8bl0hXkf5YWYNz", "name": "generator.pt"},
    "slamdunk-N": {"id": "1-_L7YVb48sLr_kPpOcn4dUq7Cv08WQuG", "name": "sampler.pt"},
    "slamdunk-S": {"id": "1Dgh11ZeXS2XIV2eJZAExWMjogxi_m_C8", "name": "exstyle_code.npy"},
}


def prepare_models():
    # download pSp encoder
    path = MODEL_PATHS["encoder"]
    download_command = get_download_model_command(file_id=path["id"], file_name=path["name"])
    logger.info("Downloading pSp encoder...")
    os.system(download_command)

    # download dualstylegan
    path = MODEL_PATHS[style_type + '-G']
    download_command = get_download_model_command(file_id=path["id"], file_name=os.path.join(style_type, path["name"]))
    logger.info("Downloading dualstylegan generator...")
    os.system(download_command)

    # download sampler
    path = MODEL_PATHS[style_type + '-N']
    download_command = get_download_model_command(file_id=path["id"], file_name=os.path.join(style_type, path["name"]))
    logger.info("Downloading dualstylegan sampler...")
    os.system(download_command)

    # download extrinsic style code
    path = MODEL_PATHS[style_type + '-S']
    download_command = get_download_model_command(file_id=path["id"], file_name=os.path.join(style_type, path["name"]))
    logger.info("Downloading extrinsic style code...")
    os.system(download_command)

    # download dlib face detector
    modelname = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        logger.info("Downloading dlib's landmark predictor...")
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname + '.bz2')
        zipfile = bz2.BZ2File(modelname + '.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data)


@dataclass
class TransferEvent:
    type: str
    data: PIL.Image.Image


class Model:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # load DualStyleGAN
        generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
        generator.eval()
        ckpt = torch.load(os.path.join(MODEL_DIR, style_type, 'generator.pt'),
                          map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g_ema"])
        self.generator = generator.to(device)

        # load encoder
        model_path = os.path.join(MODEL_DIR, 'encoder.pt')
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        opts.device = device
        encoder = pSp(opts)
        encoder.eval()
        self.encoder = encoder.to(device)

        # load extrinsic style code
        self.exstyles = np.load(os.path.join(MODEL_DIR, style_type, MODEL_PATHS[style_type + '-S']["name"]),
                                allow_pickle=True).item()

        self.face_predictor = dlib.shape_predictor(DLIB_FACE_MODEL_NAME)

        logger.info('Model successfully loaded!')

    def transfer(self, image_path: str):
        def run_alignment(image_path: str) -> PIL.Image.Image:
            aligned_image = align_face(filepath=image_path, predictor=self.face_predictor)
            return aligned_image

        logger.info("Aligning image...")
        I = self.transform(run_alignment(image_path)).unsqueeze(dim=0).to(device)
        yield TransferEvent('align', I[0].cpu())

        logger.info("Start style transfer")
        style_id = 26

        # try to load the style image
        logger.info("Loading style image...")
        stylename = list(self.exstyles.keys())[style_id]
        stylepath = os.path.join(DATA_DIR, style_type, 'images/train', stylename)
        logger.debug('loading %s' % stylepath)
        if os.path.exists(stylepath):
            S = load_image(stylepath)
            yield TransferEvent('target_style', S[0])
        else:
            logger.error('%s is not found' % stylename)
            return

        logger.info("Style transfer...")
        with torch.no_grad():
            img_rec, instyle = self.encoder(I, randomize_noise=False, return_latents=True,
                                            z_plus_latent=True, return_z_plus_latent=True, resize=False)
            img_rec = torch.clamp(img_rec.detach(), -1, 1)
            logger.debug(f"reconstruction size: {len(img_rec)}")
            yield TransferEvent('reconstruction', img_rec[0].cpu())

            latent = torch.tensor(self.exstyles[stylename]).repeat(2, 1, 1).to(device)
            # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
            latent[1, 7:18] = instyle[0, 7:18]
            exstyle = self.generator.generator.style(
                latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
                latent.shape)

            img_gen, _ = self.generator([instyle.repeat(2, 1, 1)], exstyle, z_plus_latent=True,
                                        truncation=0.7, truncation_latent=0, use_res=True,
                                        interp_weights=[0.6] * 7 + [1] * 11)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            logger.debug(f"generated size: {len(img_gen)}")
            yield TransferEvent('style_transfer', img_gen[0].cpu())
            yield TransferEvent('structure_transfer', img_gen[1].cpu())
            # deactivate color-related layers by setting w_c = 0
            img_gen2, _ = self.generator([instyle], exstyle[0:1], z_plus_latent=True,
                                         truncation=0.7, truncation_latent=0, use_res=True,
                                         interp_weights=[0.6] * 7 + [0] * 11)
            img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)
            logger.debug(f"generated size: {len(img_gen2)}")
            yield TransferEvent('structure_transfer_alter', img_gen2[0].cpu())

        logger.info("Done.")
