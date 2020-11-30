from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
import PIL.Image as Image
import sys
sys.path[0] = '/disk2/mycode/mmdetection'


from mmdet import __version__
from mmdet.apis import init_detector, inference_detector, show_result_pyplot, get_Image_ready

class PersonTimeData(Dataset): #继承Dataset
    def __init__(self, model, img_dir='/disk2/mycode/common_data/person_train_dog/train', ): #__init__是初始化该类的一些基础参数
        self.img_dir = img_dir   

        self.images = os.listdir(self.img_dir)
        self.images = self.images[:1]
        self.boxes = []

        for image_name in self.images:
            img_path = os.path.join(self.img_dir, image_name)


            save_path = os.path.join('/disk2/mycode/common_data/person_train_dog/label_by_frcn', image_name.split('.')[0]+'.npy')
            if os.path.exists(save_path):
                result_person = np.load(save_path)        
            else:
                result = inference_detector(model, img_path)[0]
                result_person = result[0]
                np.save(save_path, result_person)
            # x1,x2,y1,y2
            result_person = result_person[result_person[:,4]>0.3]

            assert result_person.shape[0] != 0
                
            if result_person.shape[0]>1:
                
                result_person_ext = np.insert(result_person, 0, values=0, axis=1)
                result_person_ext[:, 0] = (result_person[:,2]- result_person[:,0])*(result_person[:,3]- result_person[:,1])
                result_person_ext = result_person_ext[(-result_person_ext[:,0]).argsort()]
                result_person = result_person_ext[0, 1:]
                result_person = np.expand_dims(result_person, 0)

            self.boxes.append(result_person)

            # if result_person.shape[0]>1:
            #     show_result_pyplot(model, img_path, result, out_file='22__.png')
            #     print(result_person.shape[0])
            #     print()



            '''
            # 保存
            import numpy as np
            a=np.array(a)
            np.save('a.npy',a)   # 保存为.npy格式
            # 读取
            a=np.load('a.npy')
            a=a.tolist()
            '''

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image_name = self.images[index] 
        # image_name = '0.png'
        img_path = os.path.join(self.img_dir, image_name)
        img = Image.open(img_path).convert('RGB')
        img = transforms.Resize((500, 500))(img)
        img = transforms.ToTensor()(img)

        person_box = self.boxes[index]

        person_box = person_box/600*500

        return img, person_box

if __name__=='__main__':
    import os
    import sys
    # os.chdir('./')
    print(os.getcwd())
    print(sys.path)
    print(os.path.exists('22.png'))
    

    config_file = './configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '/home/user/mycode/common_data/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    Faster_RCNN = init_detector(config_file, checkpoint_file, device='cpu').cuda()

    data = PersonTimeData(model=Faster_RCNN)
    dataloader = DataLoader(data, batch_size=4, shuffle=False) #使用DataLoader加载数据
    for i_batch,batch_data in enumerate(dataloader):
        img, person_box = batch_data
        print(i_batch)#打印batch编号
        if i_batch == 2:
            pass

