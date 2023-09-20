DATASET_PATHS = {
	'FFHQ': '/n/fs/visualai-scr/Data/CelebA-HQ/',
	'CelebA_HQ': '/n/fs/visualai-scr/Data/CelebA-HQ/',
	'AFHQ': '/n/fs/visualai-scr/Data/AFHQ-Dog/',
	'LSUN':  '/n/fs/yz-diff/dataset/',
    'IMAGENET': 'data/imagenet/',
}

MODEL_PATHS = {
	'AFHQ': "pretrained/afhqdog_p2.pt",
	'FFHQ': "pretrained/ffhq_10m.pt",
	'ir_se50': 'pretrained/model_ir_se50.pth',
    'IMAGENET': "pretrained/512x512_diffusion.pt",
	'shape_predictor': "pretrained/shape_predictor_68_face_landmarks.dat.bz2",
}


HYBRID_MODEL_PATHS = [
	'./checkpoint/human_face/curly_hair_t401.pth',
	'./checkpoint/human_face/with_makeup_t401.pth',
]

HYBRID_CONFIG = \
	{ 300: [0.4, 0.6, 0],
	    0: [0.15, 0.15, 0.7]}