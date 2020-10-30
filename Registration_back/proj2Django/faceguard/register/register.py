import os
import torch.utils.data
# from core import model
# import lib.basemodel as model
# from dataloader.LFW_loader import LFW
# from lib.loader import LFW
from .lib.loader import LFW
from .basemodel import *

# from ipdb import set_trace

class register():

    def __init__(self, resume, gpu=True):
        self.net = MobileFacenet()
        self.loc = 'cpu'
        if gpu:
            self.net = self.net.cuda()
            self.loc = 'cuda'
        if resume:
            ckpt = torch.load(resume, map_location=self.loc)
            self.net.load_state_dict(ckpt['net_state_dict'])

        self.net.eval()
        
    def run(self, pkl_path, img_path, gpu=True):
        if os.path.isfile(pkl_path):
            # result = scipy.io.loadmat(pkl_path)
            db = torch.load(pkl_path)
            # featureRs = result['fr']
            rfeat_list = db['feat']
            name_list = db['name']
            # nr = result['nr']
        else:
            # featureRs = None # image feature
            # nr = None # image name
            rfeat_list = None
            name_list = None


        nr_new = [img_path]
        nl_new = [img_path]
        lfw_dataset = LFW(nl_new, nr_new)
        lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=1,
                                              shuffle=False, num_workers=8, drop_last=False)    

        # count = 0
        for data in lfw_loader:
            if gpu:
                for i in range(len(data)):
                    data[i] = data[i] if self.loc == 'cpu' else data[i].cuda() # len(data)
            # count += data[0].size(0) # N
        
            data = data[2:] # only forward [2:]
            res = [self.net(d).data.cpu() for d in data]
            # TODO: transform to tensor operation
            # set_trace()
            rfeat = torch.cat([res[0], res[1]], dim=1)

            if rfeat_list is None:
                rfeat_list = rfeat
                name_list = [img_path]
            else:
                rfeat_list = torch.cat([rfeat_list, rfeat], dim=0)
                name_list.append(img_path)
            
            db = {'feat': rfeat_list,'name': name_list}
            torch.save(db, pkl_path)

        #     # featureR = np.concatenate((res[0], res[1]), 1) # imgr + imgr(horizon flip) features
        #     nr_new = np.array([nr_new])
        
        #     if featureRs is None:
        #         featureRs = featureR
        #     else:
        #         featureRs = np.concatenate((featureRs, featureR), 0) # concat with features before
        #     if nr is None:
        #         nr = nr_new
        #     else:
        #         nr = np.concatenate((nr, nr_new), 0)
        

        # result = {'fr': featureRs,'nr': nr}
        # # scipy.io.savemat(root, result) # ? torch.save
        # torch.save(result, pkl_path)



