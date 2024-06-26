import os
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat, reduce
import glob
from PIL import Image
import random

#if you use OOD_new,try this：
# class DADA2KS(Dataset):
#     def __init__(self, root_path, interval,phase,
#                   data_aug=False):
#         self.root_path = root_path
#         self.interval = interval
#         # self.transforms = transforms
#         self.data_aug = data_aug
#         self.fps = 30
#         self.phase=phase
#         self.data_list, self.end, self.NC_text, self. R_text ,self.P_text,self.C_text= self.get_data_list()
#
#
#
#     def get_data_list(self):
#         if self.phase =="train":
#             list_file = os.path.join(self.root_path+"/"+'OOD_new.txt')
#         # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
#             assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
#             fileIDs,end,NC_text,R_text,P_text,C_text= [],[],[],[],[],[]
#
#             with open(list_file, 'r',encoding='utf-8') as f:
#                 # for ids, line in enumerate(f.readlines()):
#                 for ids, line in enumerate(f.readlines()):
#                     # print(line)
#                     parts = line.strip().split('，')
#                     if len(parts) == 2:
#                         ID, end_number = parts[0].split(' ')
#                         fileIDs.append(ID)
#                         end.append(end_number)
#                         subparts = parts[1].split('//')
#                         if len(subparts) == 4:
#                             NC_text.append(subparts[0])
#                             R_text.append(subparts[1])
#                             P_text.append(subparts[2])
#                             C_text.append(subparts[3])
#             return fileIDs,end,NC_text,R_text,P_text,C_text
#
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def pross_video_data(self,video):
#          video_datas=[]
#          for fid in range(len(video)):
#              video_data=video[fid]
#              video_data=Image.open(video_data)
#              video_data = video_data.resize((224, 224))
#              video_data= np.asarray(video_data, np.float32)
#              video_datas.append(video_data)
#          video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
#          video_data = rearrange(video_data, 'f w h c -> f c w h')
#          return video_data
#
#     def read_rgbvideo(self, video_file,end):
#         """Read video frames
#         """
#         # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
#         # get the video data
#         nv=video_file[0:16]
#         rv=video_file[end-32:end-16]
#         rv_reverse = rv[::-1]
#         av=video_file[end-16:end]
#         nv=self.pross_video_data(nv)
#         rv=self.pross_video_data(rv)
#         rv_reverse=self.pross_video_data(rv_reverse)
#         av=self.pross_video_data(av)
#         return nv,rv,rv_reverse,av
#
#
#     def gather_info(self, index):
#         # # accident_id = int(self.data_list[index].split('/')[0])
#         # accident_id =self.data_list[index]
#         end=int(self.end[index])
#         NC_text= self.NC_text[index]
#         R_text= self.R_text[index]
#         P_text= self.P_text[index]
#         C_text=self.C_text[index]
#         return end,NC_text,R_text, P_text, C_text
#
#
#     def __getitem__(self, index):
#
#         end, NC_text, R_text, P_text, C_text=self.gather_info(index)
#         video_path = os.path.join(self.root_path,self.data_list[index]+"/"+"images")
#         video_path=glob.glob(video_path+'/'+"*.jpg")
#         video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
#         nv,rv,rv_reverse,av=self.read_rgbvideo(video_path,end)
#         example = {
#         "nv": nv / 127.5 - 1.0,
#         "rv": rv / 127.5 - 1.0,
#         "rv_reverse":rv_reverse/127.5 - 1.0,
#         "av": av / 127.5 - 1.0,
#         "N_t":NC_text,
#         "R_t":R_text,
#         "P_t":P_text,
#         "C_t":C_text
#     }
#         return example
#

#else: use 00D_train


class DADA2KS1(Dataset):
    def __init__(self, root_path, interval, phase,
                 data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase = phase
        self.data_list, self.tar, self.tai, self.tco, self.NC_text, self.R_text, self.P_text, self.C_text = self.get_data_list()

    def get_data_list(self):
        if self.phase == "train":
            list_file = os.path.join(self.root_path + "/" + 'OOD_train.txt')
            # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, tar, tai, tco, NC_text, R_text, P_text, C_text = [], [], [], [], [], [], [], []
            # samples_visited, visit_rows = [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    parts = line.strip().split('，')
                    if len(parts) == 2:
                        ID, tr, ta, tc = parts[0].split(' ')
                        fileIDs.append(ID)
                        tar.append(tr)
                        tai.append(ta)
                        tco.append(tc)
                        subparts = parts[1].split('//')
                        if len(subparts) == 4:
                            NC_text.append(subparts[0])
                            R_text.append(subparts[1])
                            P_text.append(subparts[2])
                            C_text.append(subparts[3])
            return fileIDs, tar, tai, tco, NC_text, R_text, P_text, C_text

    def __len__(self):
        return len(self.data_list)

    def pross_video_data(self, video):
        video_datas = []
        for fid in range(len(video)):
            video_data = video[fid]
            video_data = Image.open(video_data)
            video_data = video_data.resize((224, 224))
            video_data = np.asarray(video_data, np.float32)
            video_datas.append(video_data)
        video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        video_data = rearrange(video_data, 'f w h c -> f c w h')
        return video_data

    def read_rgbvideo(self, video_file, tai,tar,tco):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        nv = video_file[random.randint(0, tar - 16):][:16]
        rv = video_file[random.randint(tar, tai - 16):][:16]
        av = video_file[random.randint(tai, tco - 16):][:16]
        rv_reverse=video_file[::-1]
        nv = self.pross_video_data(nv)
        rv = self.pross_video_data(rv)
        rv_reverse=self.pross_video_data(rv_reverse)
        av = self.pross_video_data(av)
        return nv, rv,rv_reverse, av

    def gather_info(self, index):
        # # accident_id = int(self.data_list[index].split('/')[0])
        # accident_id =self.data_list[index]
        tar= int(self.tar[index])
        tai = int(self.tai[index])
        tco = int(self.tco[index])
        NC_text=self.NC_text[index]
        R_text = self.R_text[index]
        P_text = self.P_text[index]
        C_text = self.C_text[index]
        return tar, tai,tco, NC_text, R_text, P_text, C_text

    def __getitem__(self, index):

        tar, tai,tco, NC_text, R_text, P_text, C_text= self.gather_info(index)
        video_path = os.path.join(self.root_path, self.data_list[index] + "/" + "images")
        video_path = glob.glob(video_path + '/' + "*.jpg")
        video_path = sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        nv,rv,rv_reverse,av = self.read_rgbvideo(video_path, tai,tar,tco)
        example = {
            "nv": nv / 127.5 - 1.0,
            "rv": rv / 127.5 - 1.0,
            "av": av / 127.5 - 1.0,
            "N_t": NC_text,
            "R_t": R_text,
            "P_t": P_text,
            "C_t": C_text
        }
        return example















if __name__=="__main__":
    train_dataset = DADA2KS1(root_path=r"", interval=1,phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        pin_memory=True, drop_last=True)

    for id, batch in enumerate(train_dataloader):
            # print(step)
            print(batch["nv"].shape)
            print(batch["rv"].shape)
            print(batch["av"].shape)
            print(batch["N_t"])
            print(batch["R_t"])
            print(batch["P_t"])
            print(batch["C_t"])
