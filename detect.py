import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('your_weight.pt') # select your model.pt path
    model.predict(source='your_data.jpg',
                  conf=0.25,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  show_labels=False,
                  show_conf=False
                  # visualize=True # visualize model features maps
                  )