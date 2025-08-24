import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('your_weight.pt')
    model.val(data='your_data.yaml',
              split='val',
              imgsz=640,
              batch=8,
              device='2',
              save_json=False, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )