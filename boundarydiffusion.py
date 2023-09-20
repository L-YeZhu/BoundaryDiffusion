import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from sklearn import svm
import pickle
import torch.optim as optim

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment
from utils.distance_utils import euclidean_distance, cosine_similarity



def compute_radius(x):
    x = torch.pow(x, 2)
    r = torch.sum(x)
    r = torch.sqrt(r)
    return r




class BoundaryDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]


    def unconditional(self):
        print(self.args.exp)

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()

        # ----------- Precompute Latents -----------#
        seq_inv = np.linspace(0, 1, 999) * 999
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        ###---- boundaries---####
        # ---------- Load boundary ----------#
        classifier = pickle.load(open('./boundary/smile_boundary_h.sav', 'rb'))
        a = classifier.coef_.reshape(1, 512*8*8).astype(np.float32)
        # a = a / np.linalg.norm(a)

        z_classifier = pickle.load(open('./boundary/smile_boundary_z.sav', 'rb'))
        z_a = z_classifier.coef_.reshape(1, 3*256*256).astype(np.float32)
        z_a = z_a / np.linalg.norm(z_a) # normalized boundary                 

        x_lat = torch.randn(1, 3, 256, 256, device=self.device)
        n = 1
        print("get the sampled latent encodings x_T!")

        with torch.no_grad():
            with tqdm(total=len(seq_inv), desc=f"Generative process") as progress_bar:
                for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                    t = (torch.ones(n) * i).to(self.device)
                    t_next = (torch.ones(n) * j).to(self.device)
                    # print("check t and t_next:", t, t_next)
                    if t == self.args.t_0:
                        break
                    x_lat, h_lat = denoising_step(x_lat, t=t, t_next=t_next, models=model,
                                       logvars=self.logvar,
                                       # sampling_type=self.args.sample_type,
                                       sampling_type='ddim',
                                       b=self.betas,
                                       eta=0.0,
                                       learn_sigma=learn_sigma,
                                       )

                    progress_bar.update(1)




            # ----- Editing space ------ #
            start_distance = self.args.start_distance 
            end_distance = self.args.end_distance
            edit_img_number = self.args.edit_img_number
            linspace = np.linspace(start_distance, end_distance, edit_img_number)
            latent_code = h_lat.cpu().view(1,-1).numpy()
            linspace = linspace - latent_code.dot(a.T)
            linspace = linspace.reshape(-1, 1).astype(np.float32)
            edit_h_seq = latent_code + linspace * a


            z_linspace = np.linspace(start_distance, end_distance, edit_img_number)
            z_latent_code = x_lat.cpu().view(1,-1).numpy()
            z_linspace = z_linspace - z_latent_code.dot(z_a.T)
            z_linspace = z_linspace.reshape(-1, 1).astype(np.float32)
            edit_z_seq = z_latent_code + z_linspace * z_a             


            for k in range(edit_img_number):
                time_in_start = time.time()
                seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                seq_inv = [int(s) for s in list(seq_inv)]
                seq_inv_next = [-1] + list(seq_inv[:-1])

                with tqdm(total=len(seq_inv), desc="Generative process {}".format(it)) as progress_bar:
                    edit_h = torch.from_numpy(edit_h_seq[k]).to(self.device).view(-1, 512, 8, 8)
                    edit_z = torch.from_numpy(edit_z_seq[k]).to(self.device).view(-1, 3, 256, 256)
                    for i, j in zip(reversed(seq_inv), reversed(seq_inv_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        edit_z, edit_h = denoising_step(edit_z, t=t, t_next=t_next, models=model,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           eta = 1.0,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio,
                                           hybrid=self.args.hybrid_noise,
                                           hybrid_config=HYBRID_CONFIG,
                                           edit_h=edit_h,
                                           )

                save_edit = "unconditioned_smile_"+str(k)+".png"
                tvu.save_image((edit_z + 1) * 0.5, os.path.join("edit_output",save_edit))
                time_in_end = time.time()
                print(f"Editing for 1 image takes {time_in_end - time_in_start:.4f}s")
        return


    def radius(self):
        print(self.args.exp)

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()


        # ---------- Prepare the seq --------- #

        # seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = np.linspace(0, 1, 999) * 999
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = 1
        with torch.no_grad():
            er = 0
            x_rand = torch.randn(100, 3, 256, 256, device=self.device)
            for idx in range(100):
                x = x_rand[idx, :, :, :].unsqueeze(0)

                with tqdm(total=len(seq_inv), desc=f"Generative process") as progress_bar:
                    for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        if t == 500:
                            break
                        x, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                           logvars=self.logvar,
                                           # sampling_type=self.args.sample_type,
                                           sampling_type='ddim',
                                           b=self.betas,
                                           eta=0.0,
                                           learn_sigma=learn_sigma,
                                           )

                        progress_bar.update(1)
                    r_x = compute_radius(x)

                er += r_x
        print("Check radius at step :", er/100)


        return






    def boundary_search(self):
        print(self.args.exp)

        # ----------- Model -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset == "AFHQ":
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()


        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])


        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat, mid_h, label) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, (img, label) in enumerate(loader):
            # for step, img in enumerate(loader):

                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                label = label.to(self.config.device)

                # print("check x and label:", x.size(), label)



                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, mid_h_g = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma,
                                               # edit_h = mid_h,
                                               )

                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone(), mid_h_g.detach().clone(), label])
                    # img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone(), mid_h_g.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Training boundaries -----------#
        print("Start boundary search")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])      


        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            time_in_start = time.time()

            clf_h = svm.SVC(kernel='linear')
            clf_z = svm.SVC(kernel='linear')
            # print("clf model:",clf)

            exp_id = os.path.split(self.args.exp)[-1]
            save_name_h = f'boundary/{exp_id}_{trg_txt.replace(" ", "_")}_h.sav'
            save_name_z = f'boundary/{exp_id}_{trg_txt.replace(" ", "_")}_z.sav'
            n_train = len(img_lat_pairs_dic['train'])
            
            train_data_z = np.empty([n_train, 3*256*256])
            train_data_h = np.empty([n_train, 512*8*8])
            train_label = np.empty([n_train,],  dtype=int)


            for step, (x0, x_id, x_lat, mid_h, label) in enumerate(img_lat_pairs_dic['train']):
                train_data_h[step, :] = mid_h.view(1,-1).cpu().numpy()
                train_data_z[step, :] = x_lat.view(1,-1).cpu().numpy()
                train_label[step] = label.cpu().numpy()


            classifier_h = clf_h.fit(train_data_h, train_label)
            classifier_z = clf_z.fit(train_data_z, train_label)
            print(np.shape(train_data_h), np.shape(train_data_z), np.shape(train_label))
            # a = classifier.coef_.reshape(1, 512*8*8).astype(np.float32)
            # a = classifier.coef_.reshape(1, 3*256*256).astype(np.float32)
            # a = a / np.linalg.norm(a)
            time_in_end = time.time()
            print(f"Finding boundary takes {time_in_end - time_in_start:.4f}s")
            print("Finishing boudary seperation!")

            # boudary_save_h = 'smiling_boundary_h.sav'
            # boudary_save_z = 'smiling_boundary_z.sav'
            pickle.dump(classifier_h, open(save_name_h, 'wb'))
            pickle.dump(classifier_z, open(save_name_z, 'wb'))

            # test the accuracy ##
            n_test = len(img_lat_pairs_dic['test'])
            test_data_h = np.empty([n_test, 512*8*8])
            test_data_z = np.empty([n_test, 3*256*256])
            test_lable = np.empty([n_test,], dtype=int)
            for step, (x0, x_id, x_lat, mid_h, label) in enumerate(img_lat_pairs_dic['test']):
                test_data_h[step, :] = mid_h.view(1,-1).cpu().numpy()
                test_data_z[step, :] = x_lat.view(1,-1).cpu().numpy()
                test_lable[step] = label.cpu().numpy()
            classifier_h = pickle.load(open(save_name_h, 'rb'))
            classifier_z = pickle.load(open(save_name_z, 'rb'))
            print("Boundary loaded!")
            val_prediction_h = classifier_h.predict(test_data_h)
            val_prediction_z = classifier_z.predict(test_data_z)
            correct_num_h = np.sum(test_lable == val_prediction_h)
            correct_num_z = np.sum(test_lable == val_prediction_z)
            # print(val_prediction_h, test_lable)
            print("Validation accuracy on h and z spaces:", correct_num_h/n_test, correct_num_z/n_test)
            print("total training and testing", n_train, n_test)


        return None




    def edit_image_boundary(self):
        # ----------- Data -----------#
        n = self.args.bs_test


        if self.args.align_face and self.config.data.dataset in ["FFHQ", "CelebA_HQ"]:
            try:
                img = run_alignment(self.args.img_path, output_size=self.config.data.image_size)
            except:
                img = Image.open(self.args.img_path).convert("RGB")
        else:
            img = Image.open(self.args.img_path).convert("RGB")
        img = img.resize((self.config.data.image_size, self.config.data.image_size), Image.ANTIALIAS)
        img = np.array(img)/255
        img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0).repeat(n, 1, 1, 1)
        img = img.to(self.config.device)
        tvu.save_image(img, os.path.join(self.args.image_folder, f'0_orig.png'))
        x0 = (img - 0.5) * 2.

        # ----------- Models -----------#
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        elif self.config.data.dataset in ["FFHQ", "AFHQ", "IMAGENET"]:
            pass
        else:
            raise ValueError

        if self.config.data.dataset in ["CelebA_HQ", "LSUN"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ", "AFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()

        # ---------- Load boundary ----------#

        boundary_h = pickle.load(open('./boundary/smile_boundary_h.sav', 'rb'))
        a = boundary_h.coef_.reshape(1, 512*8*8).astype(np.float32)
        a = a / np.linalg.norm(a)

        boundary_z = pickle.load(open('./boundary/smile_boundary_z.sav', 'rb'))
        z_a = boundary_z.coef_.reshape(1, 3*256*256).astype(np.float32)
        z_a = z_a / np.linalg.norm(z_a) # normalized boundary


        print("Boundary loaded! In shape:", np.shape(a), np.shape(z_a))


        with torch.no_grad():
            #---------------- Invert Image to Latent in case of Deterministic Inversion process -------------------#
            if self.args.deterministic_inv:
                x_lat_path = os.path.join(self.args.image_folder, f'x_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                h_lat_path = os.path.join(self.args.image_folder, f'h_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}.pth')
                if not os.path.exists(x_lat_path):
                    seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
                    seq_inv = [int(s) for s in list(seq_inv)]
                    seq_inv_next = [-1] + list(seq_inv[:-1])

                    x = x0.clone()
                    with tqdm(total=len(seq_inv), desc=f"Inversion process ") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, mid_h_g = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma,
                                               ratio=0,
                                               )


                            progress_bar.update(1)
                        x_lat = x.clone()
                        h_lat = mid_h_g.clone()
                        torch.save(x_lat, x_lat_path)
                        torch.save(h_lat, h_lat_path)

                else:
                    print('Latent exists.')
                    x_lat = torch.load(x_lat_path)
                    h_lat = torch.load(h_lat_path)
            print("Finish inversion for the given image!", h_lat.size())


            # ----------- Generative Process -----------#
            print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}, "
                  f" Steps: {self.args.n_test_step}/{self.args.t_0}")


            # ----- Editing space ------ #
            start_distance = self.args.start_distance 
            end_distance = self.args.end_distance
            edit_img_number = self.args.edit_img_number
            # [-100, 100]
            linspace = np.linspace(start_distance, end_distance, edit_img_number)
            latent_code = h_lat.cpu().view(1,-1).numpy()
            linspace = linspace - latent_code.dot(a.T)
            linspace = linspace.reshape(-1, 1).astype(np.float32)
            edit_h_seq = latent_code + linspace * a


            z_linspace = np.linspace(start_distance, end_distance, edit_img_number)
            z_latent_code = x_lat.cpu().view(1,-1).numpy()
            z_linspace = z_linspace - z_latent_code.dot(z_a.T)
            z_linspace = z_linspace.reshape(-1, 1).astype(np.float32)
            edit_z_seq = z_latent_code + z_linspace * z_a           


            if self.args.n_test_step != 0:
                seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
                seq_test = [int(s) for s in list(seq_test)]
                print('Uniform skip type')
            else:
                seq_test = list(range(self.args.t_0))
                print('No skip')
            seq_test_next = [-1] + list(seq_test[:-1])      

            for it in range(self.args.n_iter):
                if self.args.deterministic_inv:
                    x = x_lat.clone()
                else:
                    e = torch.randn_like(x0)
                    a = (1 - self.betas).cumprod(dim=0)
                    x = x0 * a[self.args.t_0 - 1].sqrt() + e * (1.0 - a[self.args.t_0 - 1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'1_lat_ninv{self.args.n_inv_step}.png'))


                for k in range(edit_img_number):
                    time_in_start = time.time()

                    with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                        edit_h = torch.from_numpy(edit_h_seq[k]).to(self.device).view(-1, 512, 8, 8)
                        edit_z = torch.from_numpy(edit_z_seq[k]).to(self.device).view(-1, 3, 256, 256)
                        for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            edit_z, edit_h = denoising_step(edit_z, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               eta = 1.0,
                                               learn_sigma=learn_sigma,
                                               ratio=self.args.model_ratio,
                                               hybrid=self.args.hybrid_noise,
                                               hybrid_config=HYBRID_CONFIG,
                                               edit_h=edit_h,
                                               )


                    x0 = x.clone()
                    save_edit = "edited_"+str(k)+".png"
                    tvu.save_image((edit_z + 1) * 0.5, os.path.join("edit_output",save_edit))
                    time_in_end = time.time()
                    print(f"Editing for 1 image takes {time_in_end - time_in_start:.4f}s")
                    

                # this is for recons
                with tqdm(total=len(seq_test), desc="Generative process {}".format(it)) as progress_bar:
                    for i, j in zip(reversed(seq_test), reversed(seq_test_next)):
                        t = (torch.ones(n) * i).to(self.device)
                        t_next = (torch.ones(n) * j).to(self.device)
                        x_lat, _ = denoising_step(x_lat, t=t, t_next=t_next, models=model,
                                           logvars=self.logvar,
                                           sampling_type=self.args.sample_type,
                                           b=self.betas,
                                           # eta=self.args.eta,
                                           eta = 0.0,
                                           learn_sigma=learn_sigma,
                                           ratio=self.args.model_ratio,
                                           hybrid=self.args.hybrid_noise,
                                           hybrid_config=HYBRID_CONFIG,
                                           edit_h=None,
                                           )

                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                       f'2_lat_t{self.args.t_0}_ninv{self.args.n_inv_step}_ngen{self.args.n_test_step}_{i}_it{it}.png'))
                        progress_bar.update(1)

                x0 = x.clone()
                save_edit = "recons.png"
                tvu.save_image((x_lat + 1) * 0.5, os.path.join("edit_output",save_edit))

        return None



